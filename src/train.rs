use crate::BurnBackend;
use crate::data::{LineartBatch, LineartDataset, lineart_dataloader};
use crate::image_io::Result as AppResult;
use crate::model::{DiffusionModel, DiffusionModelConfig};
use burn::backend::{Autodiff, Flex};
use burn::config::Config;
use burn::data::dataset::Dataset;
use burn::data::dataset::transform::SelectionDataset;
use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::AdamConfig;
use burn::record::CompactRecorder;
use burn::tensor::backend::{AutodiffBackend, Backend, BackendTypes};
use burn::tensor::module::adaptive_avg_pool2d;
use burn::tensor::{Distribution, Int, Tensor, TensorData, Transaction};
use burn::train::metric::{
    Adaptor, ItemLazy, LossInput, LossMetric, Metric, MetricAttributes, MetricName, SerializedEntry,
};
use burn::train::{InferenceStep, Learner, SupervisedTraining, TrainOutput, TrainStep};
use image::codecs::png::PngEncoder;
use image::{ColorType, ImageBuffer, ImageEncoder, Luma};
use serde_json::Value;
use std::collections::HashSet;
use std::fs::File;
use std::io::{Error, ErrorKind};
use std::path::{Path, PathBuf};
use std::sync::Arc;

type TrainBackend = Autodiff<BurnBackend>;

#[derive(Config, Debug)]
pub struct TrainingConfig {
    #[config(default = "DiffusionModelConfig::new()")]
    pub model: DiffusionModelConfig,
    #[config(default = "AdamConfig::new()")]
    pub optimizer: AdamConfig,
    #[config(default = "\"dataset/images\".to_string()")]
    pub dataset_dir: String,
    #[config(default = "\"tmp/training\".to_string()")]
    pub artifact_dir: String,
    #[config(default = "1000")]
    pub num_epochs: usize,
    #[config(default = "8")]
    pub batch_size: usize,
    #[config(default = "64")]
    pub valid_count: usize,
    #[config(default = "0")]
    pub num_workers: usize,
    #[config(default = "1.0e-4")]
    pub learning_rate: f64,
    #[config(default = "42")]
    pub seed: u64,
    #[config(default = "None")]
    pub resume_epoch: Option<usize>,
    #[config(default = "true")]
    pub sample_export_enabled: bool,
    #[config(default = "\"samples\".to_string()")]
    pub sample_export_dir: String,
    #[config(default = "(0..=10).map(|index| index as f32 / 10.0).collect()")]
    pub sample_noise_levels: Vec<f32>,
    #[config(default = "20")]
    pub sample_denoising_steps: usize,
    #[config(default = "true")]
    pub balance_loss_by_tone: bool,
}

pub fn load_training_config(path: impl AsRef<Path>) -> AppResult<TrainingConfig> {
    let mut config = serde_json::to_value(TrainingConfig::new())?;
    let overrides = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    merge_json(&mut config, overrides);

    Ok(serde_json::from_value(config)?)
}

#[derive(Debug)]
pub struct SampleOutput<B: Backend> {
    noise_level: f32,
    image: Tensor<B, 2>,
}

#[derive(Debug)]
pub struct DiffusionOutput<B: Backend> {
    loss: Tensor<B, 1>,
    output: Tensor<B, 2>,
    targets: Tensor<B, 2>,
    samples: Vec<SampleOutput<B>>,
}

impl<B: Backend> DiffusionOutput<B> {
    fn new(loss: Tensor<B, 1>, output: Tensor<B, 2>, targets: Tensor<B, 2>) -> Self {
        Self {
            loss,
            output,
            targets,
            samples: Vec::new(),
        }
    }

    fn with_samples(mut self, samples: Vec<SampleOutput<B>>) -> Self {
        self.samples = samples;
        self
    }
}

impl<B: Backend> Adaptor<LossInput<B>> for DiffusionOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

impl Adaptor<SampleExportInput> for DiffusionOutput<Flex> {
    fn adapt(&self) -> SampleExportInput {
        SampleExportInput {
            samples: self
                .samples
                .iter()
                .map(|sample| SampleExportImage {
                    noise_level: sample.noise_level,
                    image: sample.image.clone(),
                })
                .collect(),
        }
    }
}

impl<B: Backend> ItemLazy for DiffusionOutput<B> {
    type ItemSync = DiffusionOutput<Flex>;

    fn sync(self) -> Self::ItemSync {
        let [output, loss, targets] = Transaction::default()
            .register(self.output)
            .register(self.loss)
            .register(self.targets)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");
        let device = &Default::default();
        let samples = self
            .samples
            .into_iter()
            .map(|sample| SampleOutput {
                noise_level: sample.noise_level,
                image: Tensor::from_data(sample.image.into_data(), device),
            })
            .collect();

        DiffusionOutput {
            loss: Tensor::from_data(loss, device),
            output: Tensor::from_data(output, device),
            targets: Tensor::from_data(targets, device),
            samples,
        }
    }
}

pub fn train_diffusion_with_config(config: TrainingConfig) -> AppResult<()> {
    let train_device: <TrainBackend as BackendTypes>::Device = Default::default();
    let valid_device: <BurnBackend as BackendTypes>::Device = Default::default();
    TrainBackend::seed(&train_device, config.seed);
    BurnBackend::seed(&valid_device, config.seed);

    std::fs::create_dir_all(&config.artifact_dir)?;
    config.save(format!("{}/config.json", config.artifact_dir))?;

    let dataset = LineartDataset::from_dir(&config.dataset_dir)?;
    assert!(
        dataset.len() > config.valid_count,
        "dataset must contain more than {} images to create a validation split; found {}",
        config.valid_count,
        dataset.len()
    );

    let selection = SelectionDataset::new_shuffled(dataset, config.seed);
    let valid_dataset = selection.slice(0, config.valid_count);
    let train_dataset = selection.slice(config.valid_count, selection.len());
    let train_loader = lineart_dataloader::<TrainBackend, _>(
        train_dataset,
        config.batch_size,
        train_device.clone(),
        Some(config.seed),
        config.num_workers,
    );
    let valid_loader =
        lineart_dataloader::<BurnBackend, _>(valid_dataset, config.batch_size, valid_device, None, config.num_workers);

    let sample_noise_levels = if config.sample_export_enabled {
        config.sample_noise_levels.clone()
    } else {
        Vec::new()
    };
    let sample_denoising_steps = if config.sample_export_enabled {
        config.sample_denoising_steps
    } else {
        0
    };
    let model = config
        .model
        .init::<TrainBackend>(&train_device)
        .with_sample_noise_levels(sample_noise_levels)
        .with_sample_denoising_steps(sample_denoising_steps)
        .with_balance_loss_by_tone(config.balance_loss_by_tone);
    let optimizer = config.optimizer.init::<TrainBackend, DiffusionModel<TrainBackend>>();
    let learner = Learner::new(model, optimizer, config.learning_rate);

    let trainer = SupervisedTraining::new(&config.artifact_dir, train_loader, valid_loader)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .num_epochs(config.num_epochs)
        .with_file_checkpointer(CompactRecorder::new())
        .summary();

    let trainer = if config.sample_export_enabled {
        trainer.metric_valid(SampleExportMetric::new(
            Path::new(&config.artifact_dir).join(&config.sample_export_dir),
        ))
    } else {
        trainer
    };

    let trainer = match config.resume_epoch {
        Some(epoch) => trainer.checkpoint(epoch),
        None => trainer,
    };

    trainer.launch(learner);

    Ok(())
}

fn merge_json(base: &mut Value, overrides: Value) {
    match (base, overrides) {
        (Value::Object(base), Value::Object(overrides)) => {
            for (key, value) in overrides {
                match base.get_mut(&key) {
                    Some(base_value) => merge_json(base_value, value),
                    None => {
                        base.insert(key, value);
                    }
                }
            }
        }
        (base, overrides) => *base = overrides,
    }
}

#[derive(Clone)]
struct SampleExportImage {
    noise_level: f32,
    image: Tensor<Flex, 2>,
}

struct SampleExportInput {
    samples: Vec<SampleExportImage>,
}

#[derive(Clone)]
struct SampleExportMetric {
    output_dir: PathBuf,
    written_epochs: HashSet<usize>,
    name: Arc<String>,
}

impl SampleExportMetric {
    fn new(output_dir: PathBuf) -> Self {
        Self {
            output_dir,
            written_epochs: HashSet::new(),
            name: Arc::new("Sample Export".to_string()),
        }
    }
}

impl Metric for SampleExportMetric {
    type Input = SampleExportInput;

    fn update(&mut self, input: &Self::Input, metadata: &burn::train::metric::MetricMetadata) -> SerializedEntry {
        let epoch = metadata.global_progress.items_processed;
        if !input.samples.is_empty() && self.written_epochs.insert(epoch) {
            if let Err(err) = std::fs::create_dir_all(&self.output_dir) {
                return SerializedEntry::new(
                    format!("sample export failed: {err}"),
                    format!("sample export failed: {err}"),
                );
            }

            let epoch_dir = self.output_dir.join(format!("epoch-{epoch:04}"));
            if let Err(err) = std::fs::create_dir_all(&epoch_dir) {
                return SerializedEntry::new(
                    format!("sample export failed: {err}"),
                    format!("sample export failed: {err}"),
                );
            }

            for sample in input.samples.iter() {
                let path = epoch_dir.join(format!("{}.png", format_noise_level(sample.noise_level)));
                if let Err(err) = write_sample_image(sample.image.clone(), &path) {
                    return SerializedEntry::new(
                        format!("sample export failed: {err}"),
                        format!("sample export failed: {err}"),
                    );
                }
            }
        }

        SerializedEntry::new("sample export".to_string(), "sample export".to_string())
    }

    fn clear(&mut self) {}

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn attributes(&self) -> MetricAttributes {
        MetricAttributes::None
    }
}

impl<B: AutodiffBackend> TrainStep for DiffusionModel<B> {
    type Input = LineartBatch<B>;
    type Output = DiffusionOutput<B>;

    fn step(&self, batch: Self::Input) -> TrainOutput<Self::Output> {
        let output = diffusion_step(self, batch.inputs);
        let grads = output.loss.clone().backward();

        TrainOutput::new(self, grads, output)
    }
}

impl<B: Backend> InferenceStep for DiffusionModel<B> {
    type Input = LineartBatch<B>;
    type Output = DiffusionOutput<B>;

    fn step(&self, batch: Self::Input) -> Self::Output {
        let output = diffusion_step(self, batch.inputs.clone());
        if self.sample_noise_levels().is_empty() {
            output
        } else {
            output.with_samples(sample_outputs(
                self,
                batch.inputs,
                self.sample_noise_levels(),
                self.sample_denoising_steps(),
            ))
        }
    }
}

fn diffusion_step<B: Backend>(model: &DiffusionModel<B>, clean: Tensor<B, 4>) -> DiffusionOutput<B> {
    let [batch_size, _, height, width] = clean.dims();
    let device = clean.device();
    let noise_level = uniform_image_noise_level(batch_size, height, width, &device);

    let clean_scales = clean_pyramid(model, clean);
    let noise_level_scales = tensor_pyramid(model, noise_level.clone());
    let mut noisy_scales = Vec::with_capacity(clean_scales.len());
    let mut matched_noises = Vec::with_capacity(clean_scales.len());

    for (clean_scale, noise_level_scale) in clean_scales.iter().zip(noise_level_scales.iter()) {
        let signal_scale = (noise_level_scale.clone().neg() + 1.0).sqrt();
        let noise_scale = noise_level_scale.clone().sqrt();
        let noise = clean_scale.random_like(Distribution::Normal(0.0, 1.0));
        let noise = match_noise_to_clean_batch(clean_scale.clone(), noise);
        let noisy = q_sample(
            clean_scale.clone(),
            noise.clone(),
            signal_scale.clone(),
            noise_scale.clone(),
        );
        noisy_scales.push(noisy);
        matched_noises.push(noise);
    }

    let noisy_original = noisy_scales[0].clone();
    let clean_original = clean_scales[0].clone();
    let noise_original = matched_noises[0].clone();
    let signal_scale = (noise_level.clone().neg() + 1.0).sqrt();
    let noise_scale = noise_level.clone().sqrt();
    let predicted_clean = model.forward(noisy_scales, noise_level);
    let predicted_v = (signal_scale.clone() * noisy_original - predicted_clean) / noise_scale.clone();
    let target_v = signal_scale * noise_original - noise_scale * clean_original;
    let loss = v_loss(
        predicted_v.clone(),
        target_v.clone(),
        clean_scales[0].clone(),
        model.balance_loss_by_tone(),
    );

    DiffusionOutput::new(
        loss,
        flatten_for_regression(predicted_v),
        flatten_for_regression(target_v),
    )
}

fn v_loss<B: Backend>(
    predicted_v: Tensor<B, 4>,
    target_v: Tensor<B, 4>,
    clean_target: Tensor<B, 4>,
    balance_loss_by_tone: bool,
) -> Tensor<B, 1> {
    if !balance_loss_by_tone {
        return MseLoss::new().forward(predicted_v, target_v, Reduction::Mean);
    }

    let squared_error = (predicted_v - target_v).square();
    (squared_error * balanced_tone_weights(clean_target)).mean()
}

fn balanced_tone_weights<B: Backend>(clean_target: Tensor<B, 4>) -> Tensor<B, 4> {
    const EPS: f64 = 1.0e-6;

    let blackness = Tensor::clamp((clean_target.neg() + 1.0) * 0.5, 0.0, 1.0);
    let whiteness = blackness.clone().neg() + 1.0;
    let black_mass = blackness.clone().mean_dim(2).mean_dim(3);
    let white_mass = whiteness.clone().mean_dim(2).mean_dim(3);
    let weight = blackness / (black_mass * 2.0 + EPS) + whiteness / (white_mass * 2.0 + EPS);

    weight.clone() / (weight.mean_dim(2).mean_dim(3) + EPS)
}

fn uniform_image_noise_level<B: Backend>(
    batch_size: usize,
    height: usize,
    width: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    Tensor::<B, 4>::random([batch_size, 1, 1, 1], Distribution::Uniform(1.0e-5, 1.0), device)
        .repeat_dim(2, height)
        .repeat_dim(3, width)
}

fn clean_pyramid<B: Backend>(model: &DiffusionModel<B>, clean: Tensor<B, 4>) -> Vec<Tensor<B, 4>> {
    tensor_pyramid(model, clean)
}

fn tensor_pyramid<B: Backend>(model: &DiffusionModel<B>, tensor: Tensor<B, 4>) -> Vec<Tensor<B, 4>> {
    let [_, _, height, width] = tensor.dims();

    model
        .input_sizes([height, width])
        .map(|size| {
            if size == [height, width] {
                tensor.clone()
            } else {
                adaptive_avg_pool2d(tensor.clone(), size)
            }
        })
        .collect()
}

fn sample_outputs<B: Backend>(
    model: &DiffusionModel<B>,
    clean: Tensor<B, 4>,
    noise_levels: &[f32],
    denoising_steps: usize,
) -> Vec<SampleOutput<B>> {
    let clean = clean.narrow(0, 0, 1);
    let noise = clean.random_like(Distribution::Normal(0.0, 1.0));
    let [_, _, height, width] = clean.dims();
    let device = clean.device();

    noise_levels
        .iter()
        .map(|&noise_level| {
            let noise_level = noise_level.clamp(0.0, 1.0);
            let noise_level_tensor = Tensor::<B, 4>::full([1, 1, height, width], noise_level, &device);
            let signal_scale = (noise_level_tensor.clone().neg() + 1.0).sqrt();
            let noise_scale = noise_level_tensor.clone().sqrt();
            let noisy = q_sample(clean.clone(), noise.clone(), signal_scale, noise_scale);
            let image = denoise_sample(model, noisy, noise_level, denoising_steps).squeeze::<2>();

            SampleOutput { noise_level, image }
        })
        .collect()
}

fn denoise_sample<B: Backend>(
    model: &DiffusionModel<B>,
    mut sample: Tensor<B, 4>,
    start_noise_level: f32,
    denoising_steps: usize,
) -> Tensor<B, 4> {
    let [_, _, height, width] = sample.dims();
    let device = sample.device();
    let schedule = denoising_schedule(start_noise_level, denoising_steps);

    for window in schedule.windows(2) {
        let noise_level = window[0];
        let next_noise_level = window[1];
        let noise_level_tensor = Tensor::<B, 4>::full([1, 1, height, width], noise_level, &device);
        let predicted_clean = model.forward(tensor_pyramid(model, sample.clone()), noise_level_tensor);

        if next_noise_level <= 0.0 || noise_level <= 1.0e-5 {
            sample = predicted_clean;
        } else {
            let signal_scale = (1.0 - noise_level).sqrt();
            let noise_scale = noise_level.sqrt();
            let next_signal_scale = (1.0 - next_noise_level).sqrt();
            let next_noise_scale = next_noise_level.sqrt();
            let predicted_noise = (sample - predicted_clean.clone() * signal_scale) / noise_scale;
            sample = predicted_clean * next_signal_scale + predicted_noise * next_noise_scale;
        }
    }

    sample
}

fn denoising_schedule(start_noise_level: f32, denoising_steps: usize) -> Vec<f32> {
    let start_noise_level = start_noise_level.clamp(0.0, 1.0);
    if start_noise_level == 0.0 {
        return vec![0.0, 0.0];
    }
    if denoising_steps == 0 {
        return vec![start_noise_level, 0.0];
    }

    (0..=denoising_steps)
        .map(|index| start_noise_level * (denoising_steps - index) as f32 / denoising_steps as f32)
        .collect()
}

fn flatten_for_regression<B: Backend>(tensor: Tensor<B, 4>) -> Tensor<B, 2> {
    let [batch_size, channels, height, width] = tensor.dims();
    tensor.reshape([batch_size, channels * height * width])
}

fn q_sample<B: Backend>(
    clean: Tensor<B, 4>,
    noise: Tensor<B, 4>,
    signal_scale: Tensor<B, 4>,
    noise_scale: Tensor<B, 4>,
) -> Tensor<B, 4> {
    clean * signal_scale + noise * noise_scale
}

fn format_noise_level(noise_level: f32) -> String {
    format!("{noise_level:.1}").replace('.', "_")
}

fn write_sample_image(tensor: Tensor<Flex, 2>, path: &Path) -> AppResult<()> {
    let data = tensor.try_into_data()?;
    let shape = data.shape.clone();
    if shape.len() != 2 {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            format!("expected a 2D image tensor, got shape {shape:?}"),
        )
        .into());
    }

    let height = shape[0];
    let width = shape[1];
    let pixels = data
        .into_vec::<f32>()?
        .into_iter()
        .map(|value| (((value.clamp(-1.0, 1.0) + 1.0) * 0.5) * 255.0).round() as u8)
        .collect::<Vec<_>>();
    let image = ImageBuffer::<Luma<u8>, Vec<u8>>::from_vec(width as u32, height as u32, pixels)
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, "tensor data length does not match shape"))?;

    let file = File::create(path)?;
    PngEncoder::new(file).write_image(image.as_raw(), width as u32, height as u32, ColorType::L8.into())?;

    Ok(())
}

fn match_noise_to_clean_batch<B: Backend>(clean: Tensor<B, 4>, noise: Tensor<B, 4>) -> Tensor<B, 4> {
    let clean_dims = clean.dims();
    let noise_dims = noise.dims();
    assert_eq!(clean_dims, noise_dims);

    let batch_size = clean_dims[0];
    if batch_size <= 1 {
        return noise;
    }

    let item_len = clean_dims[1] * clean_dims[2] * clean_dims[3];
    let clean_values = clean.detach().into_data().into_vec::<f32>().unwrap();
    let noise_values = noise.clone().detach().into_data().into_vec::<f32>().unwrap();
    let costs = pairwise_l2_costs(&clean_values, &noise_values, batch_size, item_len);
    let assignment = min_cost_assignment(&costs, batch_size)
        .into_iter()
        .map(|index| index as i32)
        .collect::<Vec<_>>();
    let indices = Tensor::<B, 1, Int>::from_data(TensorData::new(assignment, [batch_size]), &noise.device());

    noise.select(0, indices)
}

fn pairwise_l2_costs(clean: &[f32], noise: &[f32], batch_size: usize, item_len: usize) -> Vec<f64> {
    let mut costs = vec![0.0; batch_size * batch_size];

    for clean_index in 0..batch_size {
        let clean_start = clean_index * item_len;
        let clean_item = &clean[clean_start..clean_start + item_len];

        for noise_index in 0..batch_size {
            let noise_start = noise_index * item_len;
            let noise_item = &noise[noise_start..noise_start + item_len];
            costs[clean_index * batch_size + noise_index] = clean_item
                .iter()
                .zip(noise_item.iter())
                .map(|(clean, noise)| {
                    let diff = f64::from(*clean - *noise);
                    diff * diff
                })
                .sum();
        }
    }

    costs
}

fn min_cost_assignment(costs: &[f64], size: usize) -> Vec<usize> {
    assert_eq!(costs.len(), size * size);

    let mut potentials_rows = vec![0.0; size + 1];
    let mut potentials_cols = vec![0.0; size + 1];
    let mut matching_cols = vec![0usize; size + 1];
    let mut previous_cols = vec![0usize; size + 1];

    for row in 1..=size {
        matching_cols[0] = row;
        let mut col = 0usize;
        let mut min_values = vec![f64::INFINITY; size + 1];
        let mut used = vec![false; size + 1];

        loop {
            used[col] = true;
            let current_row = matching_cols[col];
            let mut delta = f64::INFINITY;
            let mut next_col = 0usize;

            for candidate_col in 1..=size {
                if used[candidate_col] {
                    continue;
                }

                let cost = costs[(current_row - 1) * size + (candidate_col - 1)]
                    - potentials_rows[current_row]
                    - potentials_cols[candidate_col];
                if cost < min_values[candidate_col] {
                    min_values[candidate_col] = cost;
                    previous_cols[candidate_col] = col;
                }
                if min_values[candidate_col] < delta {
                    delta = min_values[candidate_col];
                    next_col = candidate_col;
                }
            }

            for candidate_col in 0..=size {
                if used[candidate_col] {
                    potentials_rows[matching_cols[candidate_col]] += delta;
                    potentials_cols[candidate_col] -= delta;
                } else {
                    min_values[candidate_col] -= delta;
                }
            }

            col = next_col;
            if matching_cols[col] == 0 {
                break;
            }
        }

        loop {
            let previous_col = previous_cols[col];
            matching_cols[col] = matching_cols[previous_col];
            col = previous_col;
            if col == 0 {
                break;
            }
        }
    }

    let mut assignment = vec![0usize; size];
    for col in 1..=size {
        assignment[matching_cols[col] - 1] = col - 1;
    }

    assignment
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_cost_assignment_finds_lowest_cost_pairing() {
        let costs = [
            9.0, 1.0, 4.0, //
            6.0, 8.0, 2.0, //
            1.0, 5.0, 7.0,
        ];

        assert_eq!(min_cost_assignment(&costs, 3), vec![1, 2, 0]);
    }

    #[test]
    fn match_noise_to_clean_batch_reorders_noise_by_l2_cost() {
        let device = Default::default();
        let clean =
            Tensor::<burn::backend::Flex, 4>::from_data(TensorData::new(vec![0.0, 10.0, 20.0], [3, 1, 1, 1]), &device);
        let noise =
            Tensor::<burn::backend::Flex, 4>::from_data(TensorData::new(vec![21.0, 1.0, 9.0], [3, 1, 1, 1]), &device);

        let matched = match_noise_to_clean_batch(clean, noise);

        assert_eq!(matched.into_data().into_vec::<f32>().unwrap(), vec![1.0, 9.0, 21.0]);
    }

    #[test]
    fn uniform_image_noise_level_uses_one_value_per_image() {
        let device = Default::default();
        let noise_level = uniform_image_noise_level::<burn::backend::Flex>(2, 3, 4, &device);
        let data = noise_level.into_data().into_vec::<f32>().unwrap();

        assert_eq!(data.len(), 2 * 3 * 4);
        assert!(data[0..12].iter().all(|value| *value == data[0]));
        assert!(data[12..24].iter().all(|value| *value == data[12]));
    }

    #[test]
    fn format_noise_level_uses_filename_safe_decimal() {
        assert_eq!(format_noise_level(0.0), "0_0");
        assert_eq!(format_noise_level(0.1), "0_1");
        assert_eq!(format_noise_level(1.0), "1_0");
    }

    #[test]
    fn merge_json_keeps_unspecified_defaults() {
        let mut base = serde_json::json!({
            "batch_size": 8,
            "model": {
                "u_net": {
                    "insert_channels": 1,
                    "final_kernel_size": 3
                }
            }
        });
        let overrides = serde_json::json!({
            "model": {
                "u_net": {
                    "final_kernel_size": 5
                }
            }
        });

        merge_json(&mut base, overrides);

        assert_eq!(base["batch_size"], 8);
        assert_eq!(base["model"]["u_net"]["insert_channels"], 1);
        assert_eq!(base["model"]["u_net"]["final_kernel_size"], 5);
    }

    #[test]
    fn training_config_defaults_to_twenty_sample_denoising_steps() {
        assert_eq!(TrainingConfig::new().sample_denoising_steps, 20);
    }

    #[test]
    fn training_config_balances_loss_by_tone_by_default() {
        assert!(TrainingConfig::new().balance_loss_by_tone);
    }

    #[test]
    fn balanced_tone_weights_are_uniform_when_black_and_white_are_balanced() {
        let device = Default::default();
        let clean = Tensor::<burn::backend::Flex, 4>::from_data(
            TensorData::new(vec![-1.0, -1.0, 1.0, 1.0], [1, 1, 1, 4]),
            &device,
        );

        let weights = balanced_tone_weights(clean).into_data().into_vec::<f32>().unwrap();

        for weight in weights {
            assert!((weight - 1.0).abs() < 1.0e-4);
        }
    }

    #[test]
    fn balanced_tone_weights_increase_rare_black_pixels() {
        let device = Default::default();
        let clean = Tensor::<burn::backend::Flex, 4>::from_data(
            TensorData::new(vec![-1.0, 1.0, 1.0, 1.0], [1, 1, 1, 4]),
            &device,
        );

        let weights = balanced_tone_weights(clean).into_data().into_vec::<f32>().unwrap();

        assert!(weights[0] > weights[1]);
        assert!((weights[0] - 2.0).abs() < 1.0e-4);
        assert!((weights[1] - 2.0 / 3.0).abs() < 1.0e-4);
    }

    #[test]
    fn unbalanced_v_loss_matches_plain_mse() {
        let device = Default::default();
        let predicted = Tensor::<burn::backend::Flex, 4>::from_data(
            TensorData::new(vec![1.0, 3.0, 5.0, 7.0], [1, 1, 1, 4]),
            &device,
        );
        let target = Tensor::<burn::backend::Flex, 4>::from_data(
            TensorData::new(vec![0.0, 1.0, 2.0, 3.0], [1, 1, 1, 4]),
            &device,
        );
        let clean = Tensor::<burn::backend::Flex, 4>::from_data(
            TensorData::new(vec![-1.0, 1.0, -1.0, 1.0], [1, 1, 1, 4]),
            &device,
        );

        let loss = v_loss(predicted, target, clean, false)
            .into_data()
            .into_vec::<f32>()
            .unwrap();

        assert!((loss[0] - 7.5).abs() < 1.0e-4);
    }

    #[test]
    fn balanced_v_loss_weights_black_pixel_errors_more_when_black_is_rare() {
        let device = Default::default();
        let target = Tensor::<burn::backend::Flex, 4>::zeros([1, 1, 1, 4], &device);
        let clean = Tensor::<burn::backend::Flex, 4>::from_data(
            TensorData::new(vec![-1.0, 1.0, 1.0, 1.0], [1, 1, 1, 4]),
            &device,
        );
        let black_error = Tensor::<burn::backend::Flex, 4>::from_data(
            TensorData::new(vec![1.0, 0.0, 0.0, 0.0], [1, 1, 1, 4]),
            &device,
        );
        let white_error = Tensor::<burn::backend::Flex, 4>::from_data(
            TensorData::new(vec![0.0, 1.0, 0.0, 0.0], [1, 1, 1, 4]),
            &device,
        );

        let black_loss = v_loss(black_error, target.clone(), clean.clone(), true)
            .into_data()
            .into_vec::<f32>()
            .unwrap()[0];
        let white_loss = v_loss(white_error, target, clean, true)
            .into_data()
            .into_vec::<f32>()
            .unwrap()[0];

        assert!(black_loss > white_loss);
    }

    #[test]
    fn denoising_schedule_linearly_reaches_zero() {
        assert_eq!(
            denoising_schedule(1.0, 20),
            vec![
                1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1,
                0.05, 0.0
            ]
        );
        assert_eq!(
            denoising_schedule(0.5, 20),
            vec![
                0.5, 0.475, 0.45, 0.425, 0.4, 0.375, 0.35, 0.325, 0.3, 0.275, 0.25, 0.225, 0.2, 0.175, 0.15, 0.125,
                0.1, 0.075, 0.05, 0.025, 0.0
            ]
        );
    }

    #[test]
    fn denoising_schedule_handles_zero_steps_and_zero_noise() {
        assert_eq!(denoising_schedule(0.75, 0), vec![0.75, 0.0]);
        assert_eq!(denoising_schedule(0.0, 20), vec![0.0, 0.0]);
    }
}
