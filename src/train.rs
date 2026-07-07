use crate::BurnBackend;
use crate::data::{LineartBatch, LineartDataset, lineart_dataloader};
use crate::image_io::Result as AppResult;
use crate::model::{DiffusionModel, DiffusionModelConfig, LineartLossConfig, NoiseLevelConfig};
use burn::backend::{Autodiff, Flex};
use burn::config::Config;
use burn::data::dataset::Dataset;
use burn::data::dataset::transform::SelectionDataset;
use burn::optim::{AdamWConfig, GradientsAccumulator, GradientsParams};
use burn::record::CompactRecorder;
use burn::tensor::activation::relu;
use burn::tensor::backend::{AutodiffBackend, Backend, BackendTypes};
use burn::tensor::module::{adaptive_avg_pool2d, avg_pool2d, conv2d, max_pool2d};
use burn::tensor::ops::ConvOptions;
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
const MIN_NOISE_FOR_V_LOSS: f64 = 1.0e-3;

#[derive(Config, Debug)]
pub struct TrainingConfig {
    #[config(default = "DiffusionModelConfig::new()")]
    pub model: DiffusionModelConfig,
    #[config(default = "AdamWConfig::new()")]
    pub optimizer: AdamWConfig,
    #[config(default = "\"dataset/images\".to_string()")]
    pub dataset_dir: String,
    #[config(default = "\"tmp/training\".to_string()")]
    pub artifact_dir: String,
    #[config(default = "3000")]
    pub num_epochs: usize,
    #[config(default = "128")]
    pub batch_size: usize,
    #[config(default = "Some(256)")]
    pub noise_pool_size: Option<usize>,
    #[config(default = "16")]
    pub micro_batch_size: usize,
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
    #[config(default = "LineartLossConfig::new()")]
    pub lineart_loss: LineartLossConfig,
    #[config(default = "NoiseLevelConfig::new()")]
    pub noise_level: NoiseLevelConfig,
    #[config(default = "true")]
    pub recursive_training_insert: bool,
    #[config(default = "0.0")]
    pub scale_loss_multiplier: f64,
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
    filename: String,
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
                    filename: sample.filename.clone(),
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
                filename: sample.filename,
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
    assert!(config.batch_size > 0, "batch_size must be greater than zero");
    assert!(
        config.micro_batch_size > 0,
        "micro_batch_size must be greater than zero"
    );
    assert!(
        config.scale_loss_multiplier.is_finite() && config.scale_loss_multiplier >= 0.0,
        "scale_loss_multiplier must be finite and non-negative"
    );
    config.noise_level.validate();
    if let Some(noise_pool_size) = config.noise_pool_size {
        assert!(
            noise_pool_size >= config.batch_size,
            "noise_pool_size must be greater than or equal to batch_size"
        );
    }

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
    let valid_loader = lineart_dataloader::<BurnBackend, _>(
        valid_dataset,
        config.micro_batch_size,
        valid_device,
        None,
        config.num_workers,
    );

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
        .with_balance_loss_by_tone(config.balance_loss_by_tone)
        .with_training_batching(config.micro_batch_size, config.noise_pool_size)
        .with_lineart_loss_config(config.lineart_loss.clone())
        .with_recursive_training_insert(config.recursive_training_insert)
        .with_scale_loss_multiplier(config.scale_loss_multiplier)
        .with_noise_level_config(config.noise_level.clone());
    let optimizer = config.optimizer.init::<TrainBackend, DiffusionModel<TrainBackend>>();
    let learner = Learner::new(model, optimizer, scaled_learning_rate(&config));

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

fn scaled_learning_rate(config: &TrainingConfig) -> f64 {
    config.learning_rate * (config.batch_size as f64).sqrt()
}

#[derive(Clone)]
struct SampleExportImage {
    filename: String,
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
                let path = epoch_dir.join(format!("{}.png", sample.filename));
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
        let batch_size = batch.inputs.dims()[0];
        let matched = matched_diffusion_batch(self, batch.inputs, self.noise_pool_size(batch_size));
        let batch_size = matched.batch_size();
        let mut accumulator = GradientsAccumulator::<DiffusionModel<B>>::new();
        let mut item = None;
        let mut loss = None;

        for start in (0..batch_size).step_by(self.micro_batch_size()) {
            let micro_batch_size = (batch_size - start).min(self.micro_batch_size());
            let output = matched_diffusion_output(self, &matched, start, micro_batch_size);
            let loss_weight = micro_batch_size as f64 / batch_size as f64;
            let weighted_loss = output.loss.clone() * loss_weight;
            let grads = GradientsParams::from_grads(weighted_loss.clone().backward(), self);

            accumulator.accumulate(self, grads);
            loss = Some(match loss {
                Some(loss) => loss + weighted_loss.detach(),
                None => weighted_loss.detach(),
            });
            if item.is_none() {
                item = Some(output);
            }
        }

        let mut item = item.expect("at least one micro batch is required");
        item.loss = loss.expect("at least one micro batch is required");

        TrainOutput {
            grads: accumulator.grads(),
            item,
        }
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
    let matched = matched_diffusion_batch(model, clean, 0);

    matched_diffusion_output(model, &matched, 0, matched.batch_size())
}

struct MatchedDiffusionBatch<B: Backend> {
    clean_scales: Vec<Tensor<B, 4>>,
    noise_level: Tensor<B, 4>,
    noise_level_scales: Vec<Tensor<B, 4>>,
    noisy_scales: Vec<Tensor<B, 4>>,
    matched_noises: Vec<Tensor<B, 4>>,
}

impl<B: Backend> MatchedDiffusionBatch<B> {
    fn batch_size(&self) -> usize {
        self.clean_scales[0].dims()[0]
    }
}

fn matched_diffusion_batch<B: Backend>(
    model: &DiffusionModel<B>,
    clean: Tensor<B, 4>,
    noise_pool_size: usize,
) -> MatchedDiffusionBatch<B> {
    let [batch_size, _, _, _] = clean.dims();
    let device = clean.device();
    let noise_level = random_image_noise_level(clean.clone(), model.noise_level_config());
    let noise_pool_size = if noise_pool_size == 0 {
        batch_size
    } else {
        noise_pool_size.max(batch_size)
    };

    let clean_scales = clean_pyramid(model, clean);
    let noise_level_scales = tensor_pyramid(model, noise_level.clone());
    let mut noisy_scales = Vec::with_capacity(clean_scales.len());
    let mut matched_noises = Vec::with_capacity(clean_scales.len());

    for (clean_scale, noise_level_scale) in clean_scales.iter().zip(noise_level_scales.iter()) {
        let [_, channels, height, width] = clean_scale.dims();
        let signal_scale = (noise_level_scale.clone().neg() + 1.0).sqrt();
        let noise_scale = noise_level_scale.clone().sqrt();
        let noise = Tensor::<B, 4>::random(
            [noise_pool_size, channels, height, width],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let noise =
            match_noise_pool_to_clean_cost(clean_scale.clone().detach(), noise_level_scale.clone().detach(), noise);
        let noisy = q_sample(
            clean_scale.clone(),
            noise.clone(),
            signal_scale.clone(),
            noise_scale.clone(),
        );
        noisy_scales.push(noisy);
        matched_noises.push(noise);
    }

    MatchedDiffusionBatch {
        clean_scales,
        noise_level,
        noise_level_scales,
        noisy_scales,
        matched_noises,
    }
}

fn matched_diffusion_output<B: Backend>(
    model: &DiffusionModel<B>,
    matched: &MatchedDiffusionBatch<B>,
    start: usize,
    batch_size: usize,
) -> DiffusionOutput<B> {
    let clean_scales = narrow_scales(&matched.clean_scales, start, batch_size);
    let noise_level_scales = narrow_scales(&matched.noise_level_scales, start, batch_size);
    let noisy_scales = narrow_scales(&matched.noisy_scales, start, batch_size);
    let matched_noises = narrow_scales(&matched.matched_noises, start, batch_size);
    let noise_level = matched.noise_level.clone().narrow(0, start, batch_size);
    let predicted_clean_scales =
        model.forward_training(noisy_scales.clone(), noise_level.clone(), clean_scales.clone());
    let mut losses = Vec::with_capacity(clean_scales.len());
    let mut scale_weight_sum = 0.0;
    let mut predicted_v_original = None;
    let mut target_v_original = None;

    for (scale_index, (((predicted_clean, noisy), noise), (clean, noise_level))) in predicted_clean_scales
        .into_iter()
        .zip(noisy_scales.into_iter())
        .zip(matched_noises.into_iter())
        .zip(clean_scales.iter().cloned().zip(noise_level_scales.into_iter()))
        .enumerate()
    {
        let scale_weight = scale_loss_weight(scale_index, model.scale_loss_multiplier());
        let regression = diffusion_regression_loss(
            predicted_clean.clone(),
            noisy,
            noise,
            clean.clone(),
            noise_level.clone(),
            model.balance_loss_by_tone(),
        );
        if lineart_loss_enabled(model.lineart_loss_config()) {
            let lineart_loss =
                lineart_image_loss(predicted_clean, clean, noise_level.clone(), model.lineart_loss_config());
            losses.push((regression.loss + lineart_loss) * scale_weight);
        } else {
            losses.push(regression.loss * scale_weight);
        }
        scale_weight_sum += scale_weight;

        if scale_index == 0 {
            predicted_v_original = Some(regression.output);
            target_v_original = Some(regression.target);
        }
    }

    let loss = losses
        .into_iter()
        .reduce(|total, loss| total + loss)
        .expect("at least one scale is required")
        / scale_weight_sum;
    let predicted_v = predicted_v_original.expect("original scale output should exist");
    let target_v = target_v_original.expect("original scale target should exist");

    DiffusionOutput::new(
        loss,
        flatten_for_regression(predicted_v),
        flatten_for_regression(target_v),
    )
}

fn scale_loss_weight(scale_index: usize, scale_loss_multiplier: f64) -> f64 {
    scale_loss_multiplier.powi(scale_index as i32)
}

fn narrow_scales<B: Backend>(scales: &[Tensor<B, 4>], start: usize, batch_size: usize) -> Vec<Tensor<B, 4>> {
    scales
        .iter()
        .map(|scale| scale.clone().narrow(0, start, batch_size))
        .collect()
}

struct RegressionLossOutput<B: Backend> {
    loss: Tensor<B, 1>,
    output: Tensor<B, 4>,
    target: Tensor<B, 4>,
}

fn diffusion_regression_loss<B: Backend>(
    predicted_clean: Tensor<B, 4>,
    _noisy: Tensor<B, 4>,
    _noise: Tensor<B, 4>,
    clean: Tensor<B, 4>,
    noise_level: Tensor<B, 4>,
    balance_loss_by_tone: bool,
) -> RegressionLossOutput<B> {
    let loss_scale = Tensor::clamp(noise_level, MIN_NOISE_FOR_V_LOSS, 1.0);
    let x_error = (predicted_clean.clone() - clean.clone()).square();
    let squared_error = x_error / loss_scale.clone();
    let loss = if balance_loss_by_tone {
        (squared_error * balanced_tone_weights(clean.clone())).mean()
    } else {
        squared_error.mean()
    };
    let metric_scale = loss_scale.sqrt();
    let output = predicted_clean / metric_scale.clone();
    let target = clean / metric_scale;

    RegressionLossOutput { loss, output, target }
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

fn lineart_image_loss<B: Backend>(
    predicted_clean: Tensor<B, 4>,
    clean_target: Tensor<B, 4>,
    noise_level: Tensor<B, 4>,
    config: &LineartLossConfig,
) -> Tensor<B, 1> {
    if !lineart_loss_enabled(config) {
        return clean_target.mean() * 0.0;
    }

    let predicted_black = blackness(predicted_clean);
    let clean_black = blackness(clean_target);
    let prior_weight = lineart_prior_weight(noise_level.clone());
    let noise_weight = inverse_noise_weight(noise_level);
    let mut loss = predicted_black.clone().mean() * 0.0;

    if config.density_weight != 0.0 {
        loss = loss
            + multiscale_blackness_density_loss(predicted_black.clone(), clean_black.clone(), noise_weight.clone())
                * config.density_weight;
    }
    if config.edge_weight != 0.0 {
        loss = loss
            + sobel_edge_loss(predicted_black.clone(), clean_black.clone(), noise_weight.clone()) * config.edge_weight;
    }
    if config.speckle_weight != 0.0 {
        loss = loss
            + background_speckle_loss(predicted_black.clone(), clean_black.clone(), noise_weight.clone())
                * config.speckle_weight;
    }
    if config.contrast_weight != 0.0 {
        loss = loss + contrast_loss(predicted_black.clone(), noise_weight) * config.contrast_weight;
    }
    if config.support_weight != 0.0 {
        loss = loss
            + multiscale_line_support_prior_loss(predicted_black.clone(), prior_weight.clone()) * config.support_weight;
    }
    if config.direction_weight != 0.0 {
        loss = loss + directional_line_continuity_prior_loss(predicted_black, prior_weight) * config.direction_weight;
    }

    loss
}

fn lineart_loss_enabled(config: &LineartLossConfig) -> bool {
    config.density_weight != 0.0
        || config.edge_weight != 0.0
        || config.speckle_weight != 0.0
        || config.contrast_weight != 0.0
        || config.support_weight != 0.0
        || config.direction_weight != 0.0
}

fn blackness<B: Backend>(image: Tensor<B, 4>) -> Tensor<B, 4> {
    Tensor::clamp((image.neg() + 1.0) * 0.5, 0.0, 1.0)
}

fn inverse_noise_weight<B: Backend>(noise_level: Tensor<B, 4>) -> Tensor<B, 4> {
    const NOISE_WEIGHT_FLOOR: f64 = 1.0e-5;

    1.0 / Tensor::clamp(noise_level.mean_dim(2).mean_dim(3), NOISE_WEIGHT_FLOOR, 1.0)
}

fn lineart_prior_weight<B: Backend>(noise_level: Tensor<B, 4>) -> Tensor<B, 4> {
    Tensor::clamp(noise_level.mean_dim(2).mean_dim(3), 0.0, 1.0)
}

fn multiscale_line_support_prior_loss<B: Backend>(
    predicted_black: Tensor<B, 4>,
    noise_weight: Tensor<B, 4>,
) -> Tensor<B, 1> {
    const SUPPORT_KERNELS: [usize; 4] = [3, 5, 9, 17];
    const SUPPORT_THRESHOLD: f64 = 0.5;

    let [_, _, height, width] = predicted_black.dims();
    let mut best_penalty: Option<Tensor<B, 4>> = None;

    for kernel_size in SUPPORT_KERNELS {
        if kernel_size > height || kernel_size > width {
            continue;
        }

        let support = square_neighbor_support(predicted_black.clone(), kernel_size);
        let penalty = predicted_black.clone() * relu(support.neg() + SUPPORT_THRESHOLD).square();
        best_penalty = Some(match best_penalty {
            Some(best_penalty) => best_penalty.min_pair(penalty),
            None => penalty,
        });
    }

    weighted_mean(
        best_penalty.unwrap_or_else(|| predicted_black.clone() * 0.0),
        noise_weight,
    )
}

fn directional_line_continuity_prior_loss<B: Backend>(
    predicted_black: Tensor<B, 4>,
    noise_weight: Tensor<B, 4>,
) -> Tensor<B, 1> {
    const DIRECTION_KERNELS: [usize; 3] = [3, 5, 9];
    const DIRECTION_THRESHOLD: f64 = 1.0;

    let [_, _, height, width] = predicted_black.dims();
    let mut best_response: Option<Tensor<B, 4>> = None;

    for kernel_size in DIRECTION_KERNELS {
        if kernel_size > height || kernel_size > width {
            continue;
        }

        for direction in LineDirection::ALL {
            let response = directional_line_support(predicted_black.clone(), kernel_size, direction);
            best_response = Some(match best_response {
                Some(best_response) => best_response.max_pair(response),
                None => response,
            });
        }
    }

    let penalty = match best_response {
        Some(response) => predicted_black.clone() * relu(response.neg() + DIRECTION_THRESHOLD).square(),
        None => predicted_black.clone() * 0.0,
    };

    weighted_mean(penalty, noise_weight)
}

fn square_neighbor_support<B: Backend>(blackness: Tensor<B, 4>, kernel_size: usize) -> Tensor<B, 4> {
    let center = kernel_size / 2;
    let area = (kernel_size * kernel_size) as f64;

    avg_pool2d(
        blackness.clone(),
        [kernel_size, kernel_size],
        [1, 1],
        [center, center],
        true,
        false,
    ) * area
        - blackness
}

#[derive(Clone, Copy)]
enum LineDirection {
    Horizontal,
    Vertical,
    DiagonalDown,
    DiagonalUp,
}

impl LineDirection {
    const ALL: [Self; 4] = [Self::Horizontal, Self::Vertical, Self::DiagonalDown, Self::DiagonalUp];
}

fn directional_line_support<B: Backend>(
    blackness: Tensor<B, 4>,
    kernel_size: usize,
    direction: LineDirection,
) -> Tensor<B, 4> {
    let center = kernel_size / 2;
    let radius = center as isize;
    let mut support = blackness.clone() * 0.0;

    for offset in -radius..=radius {
        if offset == 0 {
            continue;
        }

        let (dy, dx) = match direction {
            LineDirection::Horizontal => (0, offset),
            LineDirection::Vertical => (offset, 0),
            LineDirection::DiagonalDown => (offset, offset),
            LineDirection::DiagonalUp => (offset, -offset),
        };
        support = support + shifted_image(blackness.clone(), dy, dx);
    }

    support
}

fn shifted_image<B: Backend>(image: Tensor<B, 4>, dy: isize, dx: isize) -> Tensor<B, 4> {
    let [batch_size, channels, height, width] = image.dims();
    let (source_y_start, destination_y_start, length_y) = shifted_ranges(height, dy);
    let (source_x_start, destination_x_start, length_x) = shifted_ranges(width, dx);
    let source = image.clone().slice([
        0..batch_size,
        0..channels,
        source_y_start..source_y_start + length_y,
        source_x_start..source_x_start + length_x,
    ]);

    (image * 0.0).slice_assign(
        [
            0..batch_size,
            0..channels,
            destination_y_start..destination_y_start + length_y,
            destination_x_start..destination_x_start + length_x,
        ],
        source,
    )
}

fn shifted_ranges(size: usize, offset: isize) -> (usize, usize, usize) {
    if offset >= 0 {
        let offset = offset as usize;
        (offset, 0, size - offset)
    } else {
        let offset = (-offset) as usize;
        (0, offset, size - offset)
    }
}

fn multiscale_blackness_density_loss<B: Backend>(
    predicted_black: Tensor<B, 4>,
    clean_black: Tensor<B, 4>,
    noise_weight: Tensor<B, 4>,
) -> Tensor<B, 1> {
    const DENSITY_WINDOWS: [usize; 3] = [4, 8, 16];

    let [_, _, height, width] = predicted_black.dims();
    let mut losses = Vec::new();

    for window in DENSITY_WINDOWS {
        if window > height || window > width {
            continue;
        }

        let predicted = avg_pool2d(
            predicted_black.clone(),
            [window, window],
            [window, window],
            [0, 0],
            false,
            false,
        );
        let clean = avg_pool2d(
            clean_black.clone(),
            [window, window],
            [window, window],
            [0, 0],
            false,
            false,
        );
        losses.push(weighted_mse(predicted, clean, noise_weight.clone()));
    }

    let loss_count = losses.len();
    losses
        .into_iter()
        .reduce(|total, loss| total + loss)
        .map(|loss| loss / loss_count as f64)
        .unwrap_or_else(|| predicted_black.mean() * 0.0)
}

fn sobel_edge_loss<B: Backend>(
    predicted_black: Tensor<B, 4>,
    clean_black: Tensor<B, 4>,
    noise_weight: Tensor<B, 4>,
) -> Tensor<B, 1> {
    let predicted_edges = sobel_edges(predicted_black);
    let clean_edges = sobel_edges(clean_black);

    weighted_mse(predicted_edges, clean_edges, noise_weight)
}

fn sobel_edges<B: Backend>(blackness: Tensor<B, 4>) -> Tensor<B, 4> {
    let device = blackness.device();
    let weight = Tensor::<B, 4>::from_data(
        TensorData::new(
            vec![
                -0.25, 0.0, 0.25, -0.5, 0.0, 0.5, -0.25, 0.0, 0.25, //
                -0.25, -0.5, -0.25, 0.0, 0.0, 0.0, 0.25, 0.5, 0.25,
            ],
            [2, 1, 3, 3],
        ),
        &device,
    );

    conv2d(blackness, weight, None, ConvOptions::new([1, 1], [1, 1], [1, 1], 1))
}

fn background_speckle_loss<B: Backend>(
    predicted_black: Tensor<B, 4>,
    clean_black: Tensor<B, 4>,
    noise_weight: Tensor<B, 4>,
) -> Tensor<B, 1> {
    const EPS: f64 = 1.0e-6;

    let dilated_clean = max_pool2d(clean_black, [3, 3], [1, 1], [1, 1], [1, 1], false);
    let background = dilated_clean.neg() + 1.0;
    let background_mass = background.clone().mean_dim(2).mean_dim(3) + EPS;
    let speckle = predicted_black.square() * background.clone() / background_mass;

    weighted_mean(speckle, noise_weight)
}

fn contrast_loss<B: Backend>(predicted_black: Tensor<B, 4>, noise_weight: Tensor<B, 4>) -> Tensor<B, 1> {
    let contrast = predicted_black.clone() * (predicted_black.neg() + 1.0) * 4.0;

    weighted_mean(contrast, noise_weight)
}

fn weighted_mse<B: Backend>(predicted: Tensor<B, 4>, target: Tensor<B, 4>, noise_weight: Tensor<B, 4>) -> Tensor<B, 1> {
    weighted_mean((predicted - target).square(), noise_weight)
}

fn weighted_mean<B: Backend>(loss_map: Tensor<B, 4>, noise_weight: Tensor<B, 4>) -> Tensor<B, 1> {
    (loss_map * noise_weight).mean()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NoiseLevelPattern {
    Full,
    Band,
    Rectangle,
    RegionLineClean,
}

#[derive(Clone, Copy)]
enum BandSide {
    Left,
    Right,
    Top,
    Bottom,
}

#[derive(Clone, Copy)]
enum NoiseRegion {
    Band {
        side: BandSide,
        span: usize,
    },
    Rectangle {
        y_start: usize,
        x_start: usize,
        height: usize,
        width: usize,
    },
}

const NOISE_RANDOMS_PER_IMAGE: usize = 8;

fn random_image_noise_level<B: Backend>(clean: Tensor<B, 4>, config: &NoiseLevelConfig) -> Tensor<B, 4> {
    let [batch_size, _, _, _] = clean.dims();
    let device = clean.device();
    let random_values = Tensor::<B, 2>::random(
        [batch_size, NOISE_RANDOMS_PER_IMAGE],
        Distribution::Uniform(0.0, 1.0),
        &device,
    )
    .into_data()
    .into_vec::<f32>()
    .unwrap();

    image_noise_level_from_random_values(clean, config, &random_values)
}

fn image_noise_level_from_random_values<B: Backend>(
    clean: Tensor<B, 4>,
    config: &NoiseLevelConfig,
    random_values: &[f32],
) -> Tensor<B, 4> {
    config.validate();
    let [batch_size, channels, height, width] = clean.dims();
    assert_eq!(channels, 1);
    assert_eq!(random_values.len(), batch_size * NOISE_RANDOMS_PER_IMAGE);
    let device = clean.device();
    let mut noise_values = vec![0.0; batch_size * height * width];
    let mut region_line_clean_values = vec![0.0; batch_size * height * width];

    for batch in 0..batch_size {
        let random = &random_values[batch * NOISE_RANDOMS_PER_IMAGE..(batch + 1) * NOISE_RANDOMS_PER_IMAGE];
        let base_noise = 1.0e-5 + random[0] * (1.0 - 1.0e-5);
        let pattern = choose_noise_level_pattern(config, random[1]);

        match pattern {
            NoiseLevelPattern::Full => fill_image(&mut noise_values, batch, height, width, base_noise),
            NoiseLevelPattern::Band => {
                let region = band_region(height, width, random[2], random[3]);
                fill_region(&mut noise_values, batch, height, width, region, base_noise);
            }
            NoiseLevelPattern::Rectangle => {
                let region = rectangle_region(height, width, random[2], random[3], random[4], random[5]);
                fill_region(&mut noise_values, batch, height, width, region, base_noise);
            }
            NoiseLevelPattern::RegionLineClean => {
                fill_image(&mut noise_values, batch, height, width, base_noise);
                let region = if random[6] < 0.5 {
                    band_region(height, width, random[2], random[3])
                } else {
                    rectangle_region(height, width, random[2], random[3], random[4], random[5])
                };
                fill_region(&mut region_line_clean_values, batch, height, width, region, 1.0);
            }
        }
    }

    let noise_level = Tensor::<B, 4>::from_data(TensorData::new(noise_values, [batch_size, 1, height, width]), &device);
    let region_line_clean = Tensor::<B, 4>::from_data(
        TensorData::new(region_line_clean_values, [batch_size, 1, height, width]),
        &device,
    );
    let line =
        Tensor::<B, 4>::zeros([batch_size, 1, height, width], &device).mask_fill(clean.detach().lower_elem(1.0), 1.0);
    let clean_mask = region_line_clean * line;

    noise_level * (clean_mask.neg() + 1.0)
}

fn choose_noise_level_pattern(config: &NoiseLevelConfig, random: f32) -> NoiseLevelPattern {
    let mut threshold = random as f64 * config.total_weight();
    if threshold < config.full_weight {
        return NoiseLevelPattern::Full;
    }
    threshold -= config.full_weight;
    if threshold < config.band_weight {
        return NoiseLevelPattern::Band;
    }
    threshold -= config.band_weight;
    if threshold < config.rectangle_weight {
        return NoiseLevelPattern::Rectangle;
    }
    NoiseLevelPattern::RegionLineClean
}

fn band_region(height: usize, width: usize, side_random: f32, span_random: f32) -> NoiseRegion {
    let side = match (side_random * 4.0).floor() as usize {
        0 => BandSide::Left,
        1 => BandSide::Right,
        2 => BandSide::Top,
        _ => BandSide::Bottom,
    };
    let axis_size = match side {
        BandSide::Left | BandSide::Right => width,
        BandSide::Top | BandSide::Bottom => height,
    };
    let span = random_span(axis_size, span_random);

    NoiseRegion::Band { side, span }
}

fn rectangle_region(
    height: usize,
    width: usize,
    width_random: f32,
    height_random: f32,
    x_random: f32,
    y_random: f32,
) -> NoiseRegion {
    let rectangle_width = random_span(width, width_random);
    let rectangle_height = random_span(height, height_random);
    let x_start = random_start(width, rectangle_width, x_random);
    let y_start = random_start(height, rectangle_height, y_random);

    NoiseRegion::Rectangle {
        y_start,
        x_start,
        height: rectangle_height,
        width: rectangle_width,
    }
}

fn random_span(size: usize, random: f32) -> usize {
    let fraction = 0.25 + random * 0.5;
    ((size as f32 * fraction).round() as usize).clamp(1, size)
}

fn random_start(size: usize, span: usize, random: f32) -> usize {
    let max_start = size.saturating_sub(span);
    ((max_start + 1) as f32 * random).floor().min(max_start as f32) as usize
}

fn fill_image(values: &mut [f32], batch: usize, height: usize, width: usize, value: f32) {
    for y in 0..height {
        for x in 0..width {
            values[noise_value_index(batch, height, width, y, x)] = value;
        }
    }
}

fn fill_region(values: &mut [f32], batch: usize, height: usize, width: usize, region: NoiseRegion, value: f32) {
    match region {
        NoiseRegion::Band { side, span } => match side {
            BandSide::Left => fill_rectangle(values, batch, height, width, 0, 0, height, span, value),
            BandSide::Right => fill_rectangle(values, batch, height, width, 0, width - span, height, span, value),
            BandSide::Top => fill_rectangle(values, batch, height, width, 0, 0, span, width, value),
            BandSide::Bottom => fill_rectangle(values, batch, height, width, height - span, 0, span, width, value),
        },
        NoiseRegion::Rectangle {
            y_start,
            x_start,
            height: region_height,
            width: region_width,
        } => fill_rectangle(
            values,
            batch,
            height,
            width,
            y_start,
            x_start,
            region_height,
            region_width,
            value,
        ),
    }
}

fn fill_rectangle(
    values: &mut [f32],
    batch: usize,
    image_height: usize,
    image_width: usize,
    y_start: usize,
    x_start: usize,
    height: usize,
    width: usize,
    value: f32,
) {
    for y in y_start..y_start + height {
        for x in x_start..x_start + width {
            values[noise_value_index(batch, image_height, image_width, y, x)] = value;
        }
    }
}

fn noise_value_index(batch: usize, height: usize, width: usize, y: usize, x: usize) -> usize {
    (batch * height + y) * width + x
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
    let device = clean.device();
    let mut outputs = Vec::new();
    let sample_kinds = [
        SampleNoiseKind::Full,
        SampleNoiseKind::Band,
        SampleNoiseKind::Rectangle,
        SampleNoiseKind::RegionLineClean,
    ];

    for &noise_level in noise_levels {
        let noise_level = noise_level.clamp(0.0, 1.0);
        for kind in sample_kinds {
            let filename = sample_filename(kind, noise_level);
            let noise_level_tensor = fixed_sample_noise_level(clean.clone(), noise_level, kind, &device);
            let signal_scale = (noise_level_tensor.clone().neg() + 1.0).sqrt();
            let noise_scale = noise_level_tensor.clone().sqrt();
            let noisy = q_sample(clean.clone(), noise.clone(), signal_scale, noise_scale);

            let image = denoise_sample(model, noisy, noise_level_tensor, noise_level, denoising_steps).squeeze::<2>();

            outputs.push(SampleOutput {
                noise_level,
                filename,
                image,
            });
        }
    }

    outputs
}

#[derive(Clone, Copy)]
enum SampleNoiseKind {
    Full,
    Band,
    Rectangle,
    RegionLineClean,
}

fn sample_filename(kind: SampleNoiseKind, noise_level: f32) -> String {
    let noise_level = format_noise_level(noise_level);
    match kind {
        SampleNoiseKind::Full => noise_level,
        SampleNoiseKind::Band => format!("band-{noise_level}"),
        SampleNoiseKind::Rectangle => format!("rectangle-{noise_level}"),
        SampleNoiseKind::RegionLineClean => format!("region-line-clean-{noise_level}"),
    }
}

fn fixed_sample_noise_level<B: Backend>(
    clean: Tensor<B, 4>,
    noise_level: f32,
    kind: SampleNoiseKind,
    device: &B::Device,
) -> Tensor<B, 4> {
    let [batch_size, channels, height, width] = clean.dims();
    assert_eq!(batch_size, 1);
    assert_eq!(channels, 1);
    let mut noise_values = vec![0.0; height * width];
    let mut region_line_clean_values = vec![0.0; height * width];

    match kind {
        SampleNoiseKind::Full => fill_image(&mut noise_values, 0, height, width, noise_level),
        SampleNoiseKind::Band => {
            let region = NoiseRegion::Band {
                side: BandSide::Right,
                span: width.div_ceil(2),
            };
            fill_region(&mut noise_values, 0, height, width, region, noise_level);
        }
        SampleNoiseKind::Rectangle => {
            let region = centered_rectangle_region(height, width);
            fill_region(&mut noise_values, 0, height, width, region, noise_level);
        }
        SampleNoiseKind::RegionLineClean => {
            fill_image(&mut noise_values, 0, height, width, noise_level);
            fill_region(
                &mut region_line_clean_values,
                0,
                height,
                width,
                centered_rectangle_region(height, width),
                1.0,
            );
        }
    }

    let noise_level = Tensor::<B, 4>::from_data(TensorData::new(noise_values, [1, 1, height, width]), device);
    let region_line_clean =
        Tensor::<B, 4>::from_data(TensorData::new(region_line_clean_values, [1, 1, height, width]), device);
    let line = Tensor::<B, 4>::zeros([1, 1, height, width], device).mask_fill(clean.detach().lower_elem(1.0), 1.0);
    let clean_mask = region_line_clean * line;

    noise_level * (clean_mask.neg() + 1.0)
}

fn centered_rectangle_region(height: usize, width: usize) -> NoiseRegion {
    let region_height = height.div_ceil(2);
    let region_width = width.div_ceil(2);

    NoiseRegion::Rectangle {
        y_start: (height - region_height) / 2,
        x_start: (width - region_width) / 2,
        height: region_height,
        width: region_width,
    }
}

fn denoise_sample<B: Backend>(
    model: &DiffusionModel<B>,
    mut sample: Tensor<B, 4>,
    start_noise_level_tensor: Tensor<B, 4>,
    start_noise_level: f32,
    denoising_steps: usize,
) -> Tensor<B, 4> {
    let schedule = denoising_schedule(start_noise_level, denoising_steps);

    for window in schedule.windows(2) {
        let noise_level = window[0];
        let next_noise_level = window[1];
        let noise_level_factor = if start_noise_level <= 1.0e-5 {
            0.0
        } else {
            noise_level / start_noise_level
        };
        let next_noise_level_factor = if start_noise_level <= 1.0e-5 {
            0.0
        } else {
            next_noise_level / start_noise_level
        };
        let noise_level_tensor = start_noise_level_tensor.clone() * noise_level_factor;
        let predicted_clean = model.forward(tensor_pyramid(model, sample.clone()), noise_level_tensor);
        sample = denoise_next_sample(
            sample,
            predicted_clean,
            start_noise_level_tensor.clone(),
            noise_level_factor,
            next_noise_level_factor,
        );
    }

    sample
}

fn denoise_next_sample<B: Backend>(
    sample: Tensor<B, 4>,
    predicted_clean: Tensor<B, 4>,
    start_noise_level_tensor: Tensor<B, 4>,
    noise_level_factor: f32,
    next_noise_level_factor: f32,
) -> Tensor<B, 4> {
    let current_noise_level = start_noise_level_tensor.clone() * noise_level_factor;
    let next_noise_level = start_noise_level_tensor * next_noise_level_factor;
    let signal_scale = (current_noise_level.neg() + 1.0).sqrt();
    let next_signal_scale = (next_noise_level.neg() + 1.0).sqrt();
    let noise_ratio = if noise_level_factor <= 0.0 {
        0.0
    } else {
        (next_noise_level_factor / noise_level_factor).clamp(0.0, 1.0).sqrt() as f64
    };
    let predicted_clean_weight = next_signal_scale - signal_scale * noise_ratio;

    sample * noise_ratio + predicted_clean * predicted_clean_weight
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

fn match_noise_pool_to_clean_cost<B: Backend>(
    clean: Tensor<B, 4>,
    noise_level: Tensor<B, 4>,
    noise: Tensor<B, 4>,
) -> Tensor<B, 4> {
    let clean_dims = clean.dims();
    let noise_dims = noise.dims();

    let batch_size = clean_dims[0];
    let noise_pool_size = noise_dims[0];
    assert_eq!(clean_dims[1], noise_dims[1]);
    assert_eq!(noise_level.dims(), clean_dims);
    assert!(noise_pool_size >= batch_size);

    let costs = pairwise_masked_noise_costs(clean.detach(), noise_level.detach(), noise.clone().detach());
    let assignment = min_cost_assignment(&costs, batch_size, noise_pool_size)
        .into_iter()
        .map(|index| index as i32)
        .collect::<Vec<_>>();
    let indices = Tensor::<B, 1, Int>::from_data(TensorData::new(assignment, [batch_size]), &noise.device());

    noise.select(0, indices)
}

fn low_frequency_cost_tensor<B: Backend>(tensor: Tensor<B, 4>) -> Tensor<B, 4> {
    const COST_SIZE: usize = 32;
    let [_, _, height, width] = tensor.dims();
    let cost_size = [height.min(COST_SIZE), width.min(COST_SIZE)];

    if cost_size == [height, width] {
        tensor
    } else {
        adaptive_avg_pool2d(tensor, cost_size)
    }
}

fn pairwise_masked_noise_costs<B: Backend>(
    clean: Tensor<B, 4>,
    noise_level: Tensor<B, 4>,
    noise: Tensor<B, 4>,
) -> Vec<f64> {
    let [clean_count, _, _, _] = clean.dims();
    let [noise_count, _, _, _] = noise.dims();
    let mut costs = vec![0.0; clean_count * noise_count];
    let signal_scale = (noise_level.clone().neg() + 1.0).sqrt();
    let noise_scale = noise_level.sqrt();

    for noise_index in 0..noise_count {
        let noise_candidate = noise.clone().narrow(0, noise_index, 1);
        let noisy = q_sample(
            clean.clone(),
            noise_candidate,
            signal_scale.clone(),
            noise_scale.clone(),
        );
        let noise_costs = low_frequency_cost_tensor((clean.clone() - noisy).square())
            .sum_dims(&[1, 2, 3])
            .into_data()
            .into_vec::<f32>()
            .unwrap();

        for (clean_index, cost) in noise_costs.into_iter().enumerate() {
            costs[clean_index * noise_count + noise_index] = f64::from(cost);
        }
    }

    costs
}

fn min_cost_assignment(costs: &[f64], row_count: usize, col_count: usize) -> Vec<usize> {
    assert!(col_count >= row_count);
    assert_eq!(costs.len(), row_count * col_count);

    let mut potentials_rows = vec![0.0; row_count + 1];
    let mut potentials_cols = vec![0.0; col_count + 1];
    let mut matching_cols = vec![0usize; col_count + 1];
    let mut previous_cols = vec![0usize; col_count + 1];

    for row in 1..=row_count {
        matching_cols[0] = row;
        let mut col = 0usize;
        let mut min_values = vec![f64::INFINITY; col_count + 1];
        let mut used = vec![false; col_count + 1];

        loop {
            used[col] = true;
            let current_row = matching_cols[col];
            let mut delta = f64::INFINITY;
            let mut next_col = 0usize;

            for candidate_col in 1..=col_count {
                if used[candidate_col] {
                    continue;
                }

                let cost = costs[(current_row - 1) * col_count + (candidate_col - 1)]
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

            for candidate_col in 0..=col_count {
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

    let mut assignment = vec![0usize; row_count];
    for col in 1..=col_count {
        if matching_cols[col] != 0 {
            assignment[matching_cols[col] - 1] = col - 1;
        }
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

        assert_eq!(min_cost_assignment(&costs, 3, 3), vec![1, 2, 0]);
    }

    #[test]
    fn min_cost_assignment_supports_more_columns_than_rows() {
        let costs = [
            9.0, 1.0, 4.0, 8.0, //
            6.0, 8.0, 2.0, 3.0,
        ];

        assert_eq!(min_cost_assignment(&costs, 2, 4), vec![1, 2]);
    }

    #[test]
    fn match_noise_pool_to_clean_cost_reorders_noise_by_l2_cost() {
        let device = Default::default();
        let clean =
            Tensor::<burn::backend::Flex, 4>::from_data(TensorData::new(vec![0.0, 10.0, 20.0], [3, 1, 1, 1]), &device);
        let noise_level = Tensor::<burn::backend::Flex, 4>::ones([3, 1, 1, 1], &device);
        let noise =
            Tensor::<burn::backend::Flex, 4>::from_data(TensorData::new(vec![21.0, 1.0, 9.0], [3, 1, 1, 1]), &device);

        let matched = match_noise_pool_to_clean_cost(clean, noise_level, noise);

        assert_eq!(matched.into_data().into_vec::<f32>().unwrap(), vec![1.0, 9.0, 21.0]);
    }

    #[test]
    fn match_noise_pool_to_clean_cost_selects_from_larger_pool() {
        let device = Default::default();
        let clean =
            Tensor::<burn::backend::Flex, 4>::from_data(TensorData::new(vec![0.0, 20.0], [2, 1, 1, 1]), &device);
        let noise_level = Tensor::<burn::backend::Flex, 4>::ones([2, 1, 1, 1], &device);
        let noise = Tensor::<burn::backend::Flex, 4>::from_data(
            TensorData::new(vec![100.0, 19.0, -1.0, 50.0], [4, 1, 1, 1]),
            &device,
        );

        let matched = match_noise_pool_to_clean_cost(clean, noise_level, noise);

        assert_eq!(matched.into_data().into_vec::<f32>().unwrap(), vec![-1.0, 19.0]);
    }

    #[test]
    fn match_noise_pool_to_clean_cost_downsamples_noise_for_costs() {
        let device = Default::default();
        let clean =
            Tensor::<burn::backend::Flex, 4>::from_data(TensorData::new(vec![0.0, 10.0], [2, 1, 1, 1]), &device);
        let noise_level = Tensor::<burn::backend::Flex, 4>::ones([2, 1, 1, 1], &device);
        let noise = Tensor::<burn::backend::Flex, 4>::from_data(
            TensorData::new(
                vec![
                    50.0, 50.0, 50.0, 50.0, //
                    9.0, 9.0, 9.0, 9.0, //
                    -1.0, -1.0, -1.0, -1.0,
                ],
                [3, 1, 2, 2],
            ),
            &device,
        );

        let matched = match_noise_pool_to_clean_cost(clean, noise_level, noise);

        assert_eq!(
            matched.into_data().into_vec::<f32>().unwrap(),
            vec![-1.0, -1.0, -1.0, -1.0, 9.0, 9.0, 9.0, 9.0]
        );
    }

    #[test]
    fn match_noise_pool_to_clean_cost_ignores_zero_noise_regions() {
        let device = Default::default();
        let clean = Tensor::<burn::backend::Flex, 4>::zeros([1, 1, 1, 2], &device);
        let noise_level =
            Tensor::<burn::backend::Flex, 4>::from_data(TensorData::new(vec![1.0, 0.0], [1, 1, 1, 2]), &device);
        let noise = Tensor::<burn::backend::Flex, 4>::from_data(
            TensorData::new(vec![0.0, 100.0, 1.0, 0.0], [2, 1, 1, 2]),
            &device,
        );

        let matched = match_noise_pool_to_clean_cost(clean, noise_level, noise);

        assert_eq!(matched.into_data().into_vec::<f32>().unwrap(), vec![0.0, 100.0]);
    }

    #[test]
    fn low_frequency_cost_tensor_downsamples_large_images() {
        let device = Default::default();
        let tensor = Tensor::<burn::backend::Flex, 4>::zeros([2, 1, 64, 48], &device);

        let cost = low_frequency_cost_tensor(tensor);

        assert_eq!(cost.dims(), [2, 1, 32, 32]);
    }

    #[test]
    fn low_frequency_cost_tensor_keeps_small_images() {
        let device = Default::default();
        let tensor = Tensor::<burn::backend::Flex, 4>::zeros([2, 1, 16, 24], &device);

        let cost = low_frequency_cost_tensor(tensor);

        assert_eq!(cost.dims(), [2, 1, 16, 24]);
    }

    #[test]
    fn full_image_noise_level_uses_one_value_per_image() {
        let device = Default::default();
        let config = noise_level_config_for_pattern(NoiseLevelPattern::Full);
        let clean = Tensor::<burn::backend::Flex, 4>::ones([2, 1, 3, 4], &device);
        let random_values = vec![0.2; 2 * NOISE_RANDOMS_PER_IMAGE];
        let noise_level = image_noise_level_from_random_values(clean, &config, &random_values);
        let data = noise_level.into_data().into_vec::<f32>().unwrap();

        assert_eq!(data.len(), 2 * 3 * 4);
        assert!(data[0..12].iter().all(|value| *value == data[0]));
        assert!(data[12..24].iter().all(|value| *value == data[12]));
    }

    #[test]
    fn band_noise_level_uses_two_values_in_selected_band() {
        let device = Default::default();
        let config = noise_level_config_for_pattern(NoiseLevelPattern::Band);
        let clean = Tensor::<burn::backend::Flex, 4>::ones([1, 1, 3, 4], &device);
        let random_values = vec![
            0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
        ];

        let data = image_noise_level_from_random_values(clean, &config, &random_values)
            .into_data()
            .into_vec::<f32>()
            .unwrap();
        let base_noise = data[0];

        assert!(base_noise > 0.0);
        for y in 0..3 {
            assert_eq!(data[y * 4], base_noise);
            assert_eq!(data[y * 4 + 1], 0.0);
            assert_eq!(data[y * 4 + 2], 0.0);
            assert_eq!(data[y * 4 + 3], 0.0);
        }
    }

    #[test]
    fn rectangle_noise_level_uses_two_values_in_selected_rectangle() {
        let device = Default::default();
        let config = noise_level_config_for_pattern(NoiseLevelPattern::Rectangle);
        let clean = Tensor::<burn::backend::Flex, 4>::ones([1, 1, 4, 4], &device);
        let random_values = vec![
            0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
        ];

        let data = image_noise_level_from_random_values(clean, &config, &random_values)
            .into_data()
            .into_vec::<f32>()
            .unwrap();
        let base_noise = data[0];

        assert!(base_noise > 0.0);
        assert_eq!(data[0], base_noise);
        assert!(data[1..].iter().all(|value| *value == 0.0));
    }

    #[test]
    fn region_line_clean_noise_level_keeps_lines_clean_inside_region() {
        let device = Default::default();
        let config = noise_level_config_for_pattern(NoiseLevelPattern::RegionLineClean);
        let clean = Tensor::<burn::backend::Flex, 4>::from_data(
            TensorData::new(
                vec![
                    -1.0, 1.0, 1.0, 1.0, //
                    1.0, 1.0, 1.0, 1.0, //
                    1.0, 1.0, 1.0, 1.0, //
                    1.0, 1.0, 1.0, 1.0,
                ],
                [1, 1, 4, 4],
            ),
            &device,
        );
        let random_values = vec![
            0.4, 0.0, 0.5, 0.5, 0.0, 0.0, 1.0, 0.0, //
        ];

        let data = image_noise_level_from_random_values(clean, &config, &random_values)
            .into_data()
            .into_vec::<f32>()
            .unwrap();
        let base_noise = data[1];

        assert_eq!(data[0], 0.0);
        assert!(base_noise > 0.0);
        assert!(data[1..].iter().all(|value| *value == base_noise));
    }

    fn noise_level_config_for_pattern(pattern: NoiseLevelPattern) -> NoiseLevelConfig {
        let mut config = NoiseLevelConfig::new();
        config.full_weight = 0.0;
        config.band_weight = 0.0;
        config.rectangle_weight = 0.0;
        config.region_line_clean_weight = 0.0;
        match pattern {
            NoiseLevelPattern::Full => config.full_weight = 1.0,
            NoiseLevelPattern::Band => config.band_weight = 1.0,
            NoiseLevelPattern::Rectangle => config.rectangle_weight = 1.0,
            NoiseLevelPattern::RegionLineClean => config.region_line_clean_weight = 1.0,
        }
        config
    }

    #[test]
    fn format_noise_level_uses_filename_safe_decimal() {
        assert_eq!(format_noise_level(0.0), "0_0");
        assert_eq!(format_noise_level(0.1), "0_1");
        assert_eq!(format_noise_level(1.0), "1_0");
    }

    #[test]
    fn sample_filename_marks_partial_noise_kind() {
        assert_eq!(sample_filename(SampleNoiseKind::Full, 0.5), "0_5");
        assert_eq!(sample_filename(SampleNoiseKind::Band, 0.5), "band-0_5");
        assert_eq!(sample_filename(SampleNoiseKind::Rectangle, 0.5), "rectangle-0_5");
        assert_eq!(
            sample_filename(SampleNoiseKind::RegionLineClean, 0.5),
            "region-line-clean-0_5"
        );
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
    fn training_config_defaults_to_micro_batching_with_larger_noise_pool() {
        let config = TrainingConfig::new();

        assert_eq!(config.micro_batch_size, 16);
        assert_eq!(config.noise_pool_size, Some(256));
    }

    #[test]
    fn training_config_defaults_to_base_learning_rate() {
        assert_eq!(TrainingConfig::new().learning_rate, 1.0e-4);
    }

    #[test]
    fn scaled_learning_rate_uses_batch_size_square_root() {
        let mut config = TrainingConfig::new();
        config.learning_rate = 1.0e-4;
        config.batch_size = 128;

        let learning_rate = scaled_learning_rate(&config);

        assert!((learning_rate - 1.0e-4 * 128.0_f64.sqrt()).abs() < 1.0e-12);
    }

    #[test]
    fn scale_loss_weight_uses_multiplier_per_smaller_scale() {
        assert_eq!(scale_loss_weight(0, 0.25), 1.0);
        assert_eq!(scale_loss_weight(1, 0.25), 0.25);
        assert_eq!(scale_loss_weight(2, 0.25), 0.0625);
        assert_eq!(scale_loss_weight(3, 2.0), 8.0);
    }

    #[test]
    fn training_config_balances_loss_by_tone_by_default() {
        assert!(TrainingConfig::new().balance_loss_by_tone);
    }

    #[test]
    fn lineart_loss_is_enabled_by_any_nonzero_weight() {
        let mut config = LineartLossConfig::new();
        config.density_weight = 0.0;
        config.edge_weight = 0.0;
        config.speckle_weight = 0.0;
        config.contrast_weight = 0.0;
        config.support_weight = 0.0;
        config.direction_weight = 0.0;

        assert!(!lineart_loss_enabled(&config));

        config.edge_weight = 0.1;

        assert!(lineart_loss_enabled(&config));

        config.edge_weight = 0.0;
        config.support_weight = 0.1;

        assert!(lineart_loss_enabled(&config));
    }

    #[test]
    fn blackness_maps_lineart_value_range() {
        let device = Default::default();
        let image =
            Tensor::<burn::backend::Flex, 4>::from_data(TensorData::new(vec![-1.0, 0.0, 1.0], [1, 1, 1, 3]), &device);

        let values = blackness(image).into_data().into_vec::<f32>().unwrap();

        assert_eq!(values, vec![1.0, 0.5, 0.0]);
    }

    #[test]
    fn contrast_loss_is_zero_for_binary_and_positive_for_gray() {
        let device = Default::default();
        let noise_weight = Tensor::<burn::backend::Flex, 4>::ones([1, 1, 1, 1], &device);
        let binary =
            Tensor::<burn::backend::Flex, 4>::from_data(TensorData::new(vec![0.0, 1.0], [1, 1, 1, 2]), &device);
        let gray = Tensor::<burn::backend::Flex, 4>::from_data(TensorData::new(vec![0.5, 0.5], [1, 1, 1, 2]), &device);

        let binary_loss = contrast_loss(binary, noise_weight.clone())
            .into_data()
            .into_vec::<f32>()
            .unwrap()[0];
        let gray_loss = contrast_loss(gray, noise_weight).into_data().into_vec::<f32>().unwrap()[0];

        assert!(binary_loss.abs() < 1.0e-6);
        assert!(gray_loss > 0.0);
    }

    #[test]
    fn support_prior_penalizes_isolated_black_more_than_line() {
        let device = Default::default();
        let noise_weight = Tensor::<burn::backend::Flex, 4>::ones([1, 1, 1, 1], &device);
        let isolated = Tensor::<burn::backend::Flex, 4>::from_data(
            TensorData::new(
                vec![
                    0.0, 0.0, 0.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 0.0, 0.0, //
                    0.0, 0.0, 1.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 0.0, 0.0,
                ],
                [1, 1, 5, 5],
            ),
            &device,
        );
        let line = Tensor::<burn::backend::Flex, 4>::from_data(
            TensorData::new(
                vec![
                    0.0, 0.0, 0.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 0.0, 0.0, //
                    1.0, 1.0, 1.0, 1.0, 1.0, //
                    0.0, 0.0, 0.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 0.0, 0.0,
                ],
                [1, 1, 5, 5],
            ),
            &device,
        );

        let isolated_loss = multiscale_line_support_prior_loss(isolated, noise_weight.clone())
            .into_data()
            .into_vec::<f32>()
            .unwrap()[0];
        let line_loss = multiscale_line_support_prior_loss(line, noise_weight)
            .into_data()
            .into_vec::<f32>()
            .unwrap()[0];

        assert!(isolated_loss > line_loss);
    }

    #[test]
    fn direction_prior_penalizes_isolated_black_more_than_line() {
        let device = Default::default();
        let noise_weight = Tensor::<burn::backend::Flex, 4>::ones([1, 1, 1, 1], &device);
        let isolated = Tensor::<burn::backend::Flex, 4>::from_data(
            TensorData::new(
                vec![
                    0.0, 0.0, 0.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 0.0, 0.0, //
                    0.0, 0.0, 1.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 0.0, 0.0,
                ],
                [1, 1, 5, 5],
            ),
            &device,
        );
        let line = Tensor::<burn::backend::Flex, 4>::from_data(
            TensorData::new(
                vec![
                    0.0, 0.0, 0.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 0.0, 0.0, //
                    1.0, 1.0, 1.0, 1.0, 1.0, //
                    0.0, 0.0, 0.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 0.0, 0.0,
                ],
                [1, 1, 5, 5],
            ),
            &device,
        );

        let isolated_loss = directional_line_continuity_prior_loss(isolated, noise_weight.clone())
            .into_data()
            .into_vec::<f32>()
            .unwrap()[0];
        let line_loss = directional_line_continuity_prior_loss(line, noise_weight)
            .into_data()
            .into_vec::<f32>()
            .unwrap()[0];

        assert!(isolated_loss > line_loss);
    }

    #[test]
    fn density_loss_is_zero_for_same_image_and_positive_for_different_density() {
        let device = Default::default();
        let noise_weight = Tensor::<burn::backend::Flex, 4>::ones([1, 1, 1, 1], &device);
        let clean = Tensor::<burn::backend::Flex, 4>::zeros([1, 1, 4, 4], &device);
        let different = Tensor::<burn::backend::Flex, 4>::ones([1, 1, 4, 4], &device);

        let same_loss = multiscale_blackness_density_loss(clean.clone(), clean.clone(), noise_weight.clone())
            .into_data()
            .into_vec::<f32>()
            .unwrap()[0];
        let different_loss = multiscale_blackness_density_loss(different, clean, noise_weight)
            .into_data()
            .into_vec::<f32>()
            .unwrap()[0];

        assert!(same_loss.abs() < 1.0e-6);
        assert!(different_loss > 0.0);
    }

    #[test]
    fn edge_loss_is_zero_for_same_image_and_positive_for_shifted_line() {
        let device = Default::default();
        let noise_weight = Tensor::<burn::backend::Flex, 4>::ones([1, 1, 1, 1], &device);
        let clean = Tensor::<burn::backend::Flex, 4>::from_data(
            TensorData::new(
                vec![
                    0.0, 0.0, 1.0, 0.0, 0.0, //
                    0.0, 0.0, 1.0, 0.0, 0.0, //
                    0.0, 0.0, 1.0, 0.0, 0.0, //
                    0.0, 0.0, 1.0, 0.0, 0.0, //
                    0.0, 0.0, 1.0, 0.0, 0.0,
                ],
                [1, 1, 5, 5],
            ),
            &device,
        );
        let shifted = Tensor::<burn::backend::Flex, 4>::from_data(
            TensorData::new(
                vec![
                    0.0, 1.0, 0.0, 0.0, 0.0, //
                    0.0, 1.0, 0.0, 0.0, 0.0, //
                    0.0, 1.0, 0.0, 0.0, 0.0, //
                    0.0, 1.0, 0.0, 0.0, 0.0, //
                    0.0, 1.0, 0.0, 0.0, 0.0,
                ],
                [1, 1, 5, 5],
            ),
            &device,
        );

        let same_loss = sobel_edge_loss(clean.clone(), clean.clone(), noise_weight.clone())
            .into_data()
            .into_vec::<f32>()
            .unwrap()[0];
        let shifted_loss = sobel_edge_loss(shifted, clean, noise_weight)
            .into_data()
            .into_vec::<f32>()
            .unwrap()[0];

        assert!(same_loss.abs() < 1.0e-6);
        assert!(shifted_loss > 0.0);
    }

    #[test]
    fn sobel_edges_respond_to_horizontal_and_vertical_lines() {
        let device = Default::default();
        let vertical = Tensor::<burn::backend::Flex, 4>::from_data(
            TensorData::new(
                vec![
                    0.0, 1.0, 0.0, //
                    0.0, 1.0, 0.0, //
                    0.0, 1.0, 0.0,
                ],
                [1, 1, 3, 3],
            ),
            &device,
        );
        let horizontal = Tensor::<burn::backend::Flex, 4>::from_data(
            TensorData::new(
                vec![
                    0.0, 0.0, 0.0, //
                    1.0, 1.0, 1.0, //
                    0.0, 0.0, 0.0,
                ],
                [1, 1, 3, 3],
            ),
            &device,
        );

        let vertical_energy = sobel_edges(vertical)
            .square()
            .mean()
            .into_data()
            .into_vec::<f32>()
            .unwrap()[0];
        let horizontal_energy = sobel_edges(horizontal)
            .square()
            .mean()
            .into_data()
            .into_vec::<f32>()
            .unwrap()[0];

        assert!(vertical_energy > 0.0);
        assert!(horizontal_energy > 0.0);
    }

    #[test]
    fn speckle_loss_penalizes_black_outside_clean_line_neighborhood() {
        let device = Default::default();
        let noise_weight = Tensor::<burn::backend::Flex, 4>::ones([1, 1, 1, 1], &device);
        let clean = Tensor::<burn::backend::Flex, 4>::from_data(
            TensorData::new(
                vec![
                    0.0, 0.0, 0.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 0.0, 0.0, //
                    0.0, 0.0, 1.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 0.0, 0.0,
                ],
                [1, 1, 5, 5],
            ),
            &device,
        );
        let near_line = clean.clone();
        let far_speckle = Tensor::<burn::backend::Flex, 4>::from_data(
            TensorData::new(
                vec![
                    1.0, 0.0, 0.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 0.0, 0.0,
                ],
                [1, 1, 5, 5],
            ),
            &device,
        );

        let near_loss = background_speckle_loss(near_line, clean.clone(), noise_weight.clone())
            .into_data()
            .into_vec::<f32>()
            .unwrap()[0];
        let far_loss = background_speckle_loss(far_speckle, clean, noise_weight)
            .into_data()
            .into_vec::<f32>()
            .unwrap()[0];

        assert!(near_loss.abs() < 1.0e-6);
        assert!(far_loss > near_loss);
    }

    #[test]
    fn lineart_loss_weights_low_noise_more_than_high_noise() {
        let device = Default::default();
        let mut config = LineartLossConfig::new();
        config.density_weight = 0.0;
        config.edge_weight = 0.0;
        config.speckle_weight = 0.0;
        config.contrast_weight = 1.0;
        let predicted = Tensor::<burn::backend::Flex, 4>::zeros([1, 1, 2, 2], &device);
        let clean = Tensor::<burn::backend::Flex, 4>::ones([1, 1, 2, 2], &device);
        let low_noise = Tensor::<burn::backend::Flex, 4>::full([1, 1, 2, 2], 0.1, &device);
        let high_noise = Tensor::<burn::backend::Flex, 4>::full([1, 1, 2, 2], 1.0, &device);

        let low_noise_loss = lineart_image_loss(predicted.clone(), clean.clone(), low_noise, &config)
            .into_data()
            .into_vec::<f32>()
            .unwrap()[0];
        let high_noise_loss = lineart_image_loss(predicted, clean, high_noise, &config)
            .into_data()
            .into_vec::<f32>()
            .unwrap()[0];

        assert!(low_noise_loss > high_noise_loss);
    }

    #[test]
    fn lineart_prior_weights_high_noise_more_than_low_noise() {
        let device = Default::default();
        let mut config = LineartLossConfig::new();
        config.density_weight = 0.0;
        config.edge_weight = 0.0;
        config.speckle_weight = 0.0;
        config.contrast_weight = 0.0;
        config.support_weight = 1.0;
        config.direction_weight = 0.0;
        let predicted = Tensor::<burn::backend::Flex, 4>::from_data(
            TensorData::new(
                vec![
                    1.0, 1.0, 1.0, //
                    1.0, -1.0, 1.0, //
                    1.0, 1.0, 1.0,
                ],
                [1, 1, 3, 3],
            ),
            &device,
        );
        let clean = Tensor::<burn::backend::Flex, 4>::ones([1, 1, 3, 3], &device);
        let low_noise = Tensor::<burn::backend::Flex, 4>::full([1, 1, 3, 3], 0.0, &device);
        let high_noise = Tensor::<burn::backend::Flex, 4>::full([1, 1, 3, 3], 1.0, &device);

        let low_noise_loss = lineart_image_loss(predicted.clone(), clean.clone(), low_noise, &config)
            .into_data()
            .into_vec::<f32>()
            .unwrap()[0];
        let high_noise_loss = lineart_image_loss(predicted, clean, high_noise, &config)
            .into_data()
            .into_vec::<f32>()
            .unwrap()[0];

        assert!(high_noise_loss > low_noise_loss);
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
    fn q_sample_keeps_zero_noise_pixels_clean() {
        let device = Default::default();
        let clean = Tensor::<burn::backend::Flex, 4>::from_data(TensorData::new(vec![2.0, 3.0], [1, 1, 1, 2]), &device);
        let noise =
            Tensor::<burn::backend::Flex, 4>::from_data(TensorData::new(vec![10.0, 20.0], [1, 1, 1, 2]), &device);
        let noise_level =
            Tensor::<burn::backend::Flex, 4>::from_data(TensorData::new(vec![0.0, 1.0], [1, 1, 1, 2]), &device);
        let signal_scale = (noise_level.clone().neg() + 1.0).sqrt();
        let noise_scale = noise_level.sqrt();

        let sampled = q_sample(clean, noise, signal_scale, noise_scale)
            .into_data()
            .into_vec::<f32>()
            .unwrap();

        assert_eq!(sampled, vec![2.0, 20.0]);
    }

    #[test]
    fn band_sample_noisy_input_keeps_zero_noise_side_clean() {
        let device = Default::default();
        let clean = Tensor::<burn::backend::Flex, 4>::from_data(
            TensorData::new(
                vec![
                    -1.0, -0.5, 0.5, 1.0, //
                    0.25, 0.5, -0.25, -0.75,
                ],
                [1, 1, 2, 4],
            ),
            &device,
        );
        let noise = Tensor::<burn::backend::Flex, 4>::from_data(TensorData::new(vec![10.0; 8], [1, 1, 2, 4]), &device);
        let noise_level = fixed_sample_noise_level(clean.clone(), 0.25, SampleNoiseKind::Band, &device);
        let signal_scale = (noise_level.clone().neg() + 1.0).sqrt();
        let noise_scale = noise_level.sqrt();

        let noisy = q_sample(clean, noise, signal_scale, noise_scale)
            .into_data()
            .into_vec::<f32>()
            .unwrap();

        assert_eq!(noisy[0], -1.0);
        assert_eq!(noisy[1], -0.5);
        assert_eq!(noisy[4], 0.25);
        assert_eq!(noisy[5], 0.5);
        assert!(noisy.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn denoise_next_sample_matches_noise_prediction_update_for_positive_noise() {
        let device = Default::default();
        let sample = Tensor::<burn::backend::Flex, 4>::from_data(TensorData::new(vec![0.7], [1, 1, 1, 1]), &device);
        let predicted_clean =
            Tensor::<burn::backend::Flex, 4>::from_data(TensorData::new(vec![0.2], [1, 1, 1, 1]), &device);
        let start_noise_level =
            Tensor::<burn::backend::Flex, 4>::from_data(TensorData::new(vec![0.25], [1, 1, 1, 1]), &device);

        let next = denoise_next_sample(
            sample.clone(),
            predicted_clean.clone(),
            start_noise_level.clone(),
            1.0,
            0.25,
        )
        .into_data()
        .into_vec::<f32>()
        .unwrap()[0];

        let signal = 0.75_f32.sqrt();
        let noise_scale = 0.25_f32.sqrt();
        let next_signal = 0.9375_f32.sqrt();
        let next_noise_scale = 0.0625_f32.sqrt();
        let predicted_noise = (0.7 - 0.2 * signal) / noise_scale;
        let expected = 0.2 * next_signal + predicted_noise * next_noise_scale;

        assert!((next - expected).abs() < 1.0e-6);
    }

    #[test]
    fn denoise_next_sample_is_continuous_at_zero_noise_level() {
        let device = Default::default();
        let sample =
            Tensor::<burn::backend::Flex, 4>::from_data(TensorData::new(vec![0.7, 0.7], [1, 1, 1, 2]), &device);
        let predicted_clean =
            Tensor::<burn::backend::Flex, 4>::from_data(TensorData::new(vec![0.2, 0.2], [1, 1, 1, 2]), &device);
        let start_noise_level =
            Tensor::<burn::backend::Flex, 4>::from_data(TensorData::new(vec![0.0, 1.0e-8], [1, 1, 1, 2]), &device);

        let next = denoise_next_sample(sample, predicted_clean, start_noise_level, 1.0, 0.25)
            .into_data()
            .into_vec::<f32>()
            .unwrap();

        assert!((next[0] - next[1]).abs() < 1.0e-5);
    }

    #[test]
    fn diffusion_regression_loss_uses_noise_floor_for_zero_noise_pixels() {
        let device = Default::default();
        let predicted_clean =
            Tensor::<burn::backend::Flex, 4>::from_data(TensorData::new(vec![1.0], [1, 1, 1, 1]), &device);
        let clean = Tensor::<burn::backend::Flex, 4>::zeros([1, 1, 1, 1], &device);
        let noisy = clean.clone();
        let noise = Tensor::<burn::backend::Flex, 4>::ones([1, 1, 1, 1], &device);
        let noise_level = Tensor::<burn::backend::Flex, 4>::zeros([1, 1, 1, 1], &device);

        let output = diffusion_regression_loss(predicted_clean, noisy, noise, clean, noise_level, false);
        let loss = output.loss.into_data().into_vec::<f32>().unwrap()[0];
        let metric_output = output.output.into_data().into_vec::<f32>().unwrap()[0];
        let metric_target = output.target.into_data().into_vec::<f32>().unwrap()[0];
        let expected_scale = (MIN_NOISE_FOR_V_LOSS as f32).sqrt();

        assert!((loss - (1.0 / MIN_NOISE_FOR_V_LOSS) as f32).abs() < 1.0e-4);
        assert!((metric_output - 1.0 / expected_scale).abs() < 1.0e-4);
        assert!(metric_target.abs() < 1.0e-6);
    }

    #[test]
    fn diffusion_regression_loss_treats_zero_and_tiny_noise_continuously() {
        let device = Default::default();
        let predicted_clean =
            Tensor::<burn::backend::Flex, 4>::from_data(TensorData::new(vec![1.0, 1.0], [1, 1, 1, 2]), &device);
        let clean = Tensor::<burn::backend::Flex, 4>::zeros([1, 1, 1, 2], &device);
        let noisy = clean.clone();
        let noise = Tensor::<burn::backend::Flex, 4>::ones([1, 1, 1, 2], &device);
        let noise_level = Tensor::<burn::backend::Flex, 4>::from_data(
            TensorData::new(vec![0.0, (MIN_NOISE_FOR_V_LOSS * 0.5) as f32], [1, 1, 1, 2]),
            &device,
        );

        let output = diffusion_regression_loss(predicted_clean, noisy, noise, clean, noise_level, false);
        let loss = output.loss.into_data().into_vec::<f32>().unwrap()[0];

        assert!((loss - (1.0 / MIN_NOISE_FOR_V_LOSS) as f32).abs() < 1.0e-4);
    }

    #[test]
    fn diffusion_step_returns_original_scale_regression_items_with_multiscale_loss() {
        let device = Default::default();
        let model = DiffusionModelConfig::new()
            .init::<burn::backend::Flex>(&device)
            .with_balance_loss_by_tone(true);
        let clean = Tensor::<burn::backend::Flex, 4>::zeros([1, 1, 128, 128], &device);

        let output = diffusion_step(&model, clean);

        assert_eq!(output.loss.dims(), [1]);
        assert_eq!(output.output.dims(), [1, 128 * 128]);
        assert_eq!(output.targets.dims(), [1, 128 * 128]);
    }

    #[test]
    fn train_step_supports_micro_batches_smaller_than_matching_batch() {
        let device = Default::default();
        let model = DiffusionModelConfig::new()
            .init::<burn::backend::Autodiff<burn::backend::Flex>>(&device)
            .with_balance_loss_by_tone(true)
            .with_training_batching(2, Some(4));
        let clean = Tensor::<burn::backend::Autodiff<burn::backend::Flex>, 4>::zeros([4, 1, 128, 128], &device);

        let output = TrainStep::step(&model, LineartBatch { inputs: clean });

        assert_eq!(output.item.loss.dims(), [1]);
        assert_eq!(output.item.output.dims(), [2, 128 * 128]);
        assert_eq!(output.item.targets.dims(), [2, 128 * 128]);
        assert!(!output.grads.is_empty());
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
