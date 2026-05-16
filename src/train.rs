use crate::BurnBackend;
use crate::data::{LineartBatch, LineartDataset, lineart_dataloader};
use crate::image_io::Result as AppResult;
use crate::model::{DiffusionModel, DiffusionModelConfig};
use burn::backend::Autodiff;
use burn::config::Config;
use burn::data::dataset::Dataset;
use burn::data::dataset::transform::SelectionDataset;
use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::AdamConfig;
use burn::record::CompactRecorder;
use burn::tensor::backend::{AutodiffBackend, Backend, BackendTypes};
use burn::tensor::module::adaptive_avg_pool2d;
use burn::tensor::{Distribution, Int, Tensor, TensorData};
use burn::train::metric::LossMetric;
use burn::train::{InferenceStep, Learner, RegressionOutput, SupervisedTraining, TrainOutput, TrainStep};

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
    #[config(default = "10")]
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

    let model = config.model.init::<TrainBackend>(&train_device);
    let optimizer = config.optimizer.init::<TrainBackend, DiffusionModel<TrainBackend>>();
    let learner = Learner::new(model, optimizer, config.learning_rate);

    let trainer = SupervisedTraining::new(&config.artifact_dir, train_loader, valid_loader)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .num_epochs(config.num_epochs)
        .with_file_checkpointer(CompactRecorder::new())
        .summary();

    let trainer = match config.resume_epoch {
        Some(epoch) => trainer.checkpoint(epoch),
        None => trainer,
    };

    trainer.launch(learner);

    Ok(())
}

impl<B: AutodiffBackend> TrainStep for DiffusionModel<B> {
    type Input = LineartBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, batch: Self::Input) -> TrainOutput<Self::Output> {
        let output = diffusion_step(self, batch.inputs);
        let grads = output.loss.clone().backward();

        TrainOutput::new(self, grads, output)
    }
}

impl<B: Backend> InferenceStep for DiffusionModel<B> {
    type Input = LineartBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, batch: Self::Input) -> Self::Output {
        diffusion_step(self, batch.inputs)
    }
}

fn diffusion_step<B: Backend>(model: &DiffusionModel<B>, clean: Tensor<B, 4>) -> RegressionOutput<B> {
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
    let loss = MseLoss::new().forward(predicted_v.clone(), target_v.clone(), Reduction::Mean);

    RegressionOutput::new(
        loss,
        flatten_for_regression(predicted_v),
        flatten_for_regression(target_v),
    )
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
}
