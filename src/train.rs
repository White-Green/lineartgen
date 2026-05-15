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
use burn::tensor::{Distribution, Tensor};
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
    let shape = clean.dims();
    let batch_size = shape[0];
    let device = clean.device();
    let noise = clean.random_like(Distribution::Normal(0.0, 1.0));
    let noise_level = Tensor::<B, 4>::random([batch_size, 1, 1, 1], Distribution::Uniform(0.0, 1.0), &device);

    let noisy = q_sample(clean, noise.clone(), noise_level.clone());
    let predicted_noise = model.forward(noisy, noise_level);
    let loss = MseLoss::new().forward(predicted_noise.clone(), noise.clone(), Reduction::Mean);

    RegressionOutput::new(
        loss,
        flatten_for_regression(predicted_noise),
        flatten_for_regression(noise),
    )
}

fn flatten_for_regression<B: Backend>(tensor: Tensor<B, 4>) -> Tensor<B, 2> {
    let [batch_size, channels, height, width] = tensor.dims();
    tensor.reshape([batch_size, channels * height * width])
}

fn q_sample<B: burn::tensor::backend::Backend>(
    clean: Tensor<B, 4>,
    noise: Tensor<B, 4>,
    noise_level: Tensor<B, 4>,
) -> Tensor<B, 4> {
    let signal_scale = (noise_level.clone().neg() + 1.0).sqrt();
    let noise_scale = noise_level.sqrt();
    clean * signal_scale + noise * noise_scale
}
