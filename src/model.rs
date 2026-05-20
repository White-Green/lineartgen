use crate::model::u_net::{UNet, UNetConfig};
use burn::config::Config;
use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::module::adaptive_avg_pool2d;
use burn::tensor::{Tensor, TensorCreationOptions};
use std::iter;

mod u_net;

#[derive(Config, Debug)]
pub struct DiffusionModelConfig {
    #[config(default = "UNetConfig::new()")]
    pub u_net: UNetConfig,
}

#[derive(Module, Debug)]
pub struct DiffusionModel<B: Backend> {
    u_net: UNet<B>,
    #[module(skip)]
    sample_noise_levels: Vec<f32>,
    #[module(skip)]
    sample_denoising_steps: usize,
    #[module(skip)]
    balance_loss_by_tone: bool,
}

impl DiffusionModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DiffusionModel<B> {
        let u_net = self.u_net.init(device);
        DiffusionModel {
            u_net,
            sample_noise_levels: Vec::new(),
            sample_denoising_steps: 0,
            balance_loss_by_tone: false,
        }
    }
}

impl<B: Backend> DiffusionModel<B> {
    pub fn with_sample_noise_levels(mut self, sample_noise_levels: Vec<f32>) -> Self {
        self.sample_noise_levels = sample_noise_levels;
        self
    }

    pub fn with_sample_denoising_steps(mut self, sample_denoising_steps: usize) -> Self {
        self.sample_denoising_steps = sample_denoising_steps;
        self
    }

    pub fn with_balance_loss_by_tone(mut self, balance_loss_by_tone: bool) -> Self {
        self.balance_loss_by_tone = balance_loss_by_tone;
        self
    }

    pub fn sample_noise_levels(&self) -> &[f32] {
        &self.sample_noise_levels
    }

    pub fn sample_denoising_steps(&self) -> usize {
        self.sample_denoising_steps
    }

    pub fn balance_loss_by_tone(&self) -> bool {
        self.balance_loss_by_tone
    }

    pub fn input_sizes(&self, base_size: [usize; 2]) -> impl Iterator<Item = [usize; 2]> {
        assert_eq!(base_size[0].next_power_of_two(), base_size[0]);
        assert_eq!(base_size[1].next_power_of_two(), base_size[1]);
        let minimum_input_size = self.u_net.minimum_input_size();
        iter::successors(Some(base_size), |&size| Some(self.u_net.expected_insert_size(size)))
            .take_while(move |&size| minimum_input_size[0] <= size[0] && minimum_input_size[1] <= size[1])
    }

    pub fn forward(&self, input: Vec<Tensor<B, 4>>, noise_level: Tensor<B, 4>) -> Tensor<B, 4> {
        assert!(
            input
                .iter()
                .map(Tensor::dims)
                .map(|[_, _, w, h]| [w, h])
                .eq(self.input_sizes([input[0].dims()[2], input[0].dims()[3]]))
        );
        let noise_levels = tensor_pyramid(self.input_sizes([input[0].dims()[2], input[0].dims()[3]]), noise_level);
        let [batches, _, w, h] = input.last().unwrap().dims();
        let [empty_h, empty_w] = self.u_net.expected_insert_size([w, h]);
        let mut a = Tensor::full(
            [batches, 1, empty_h, empty_w],
            -1.0,
            TensorCreationOptions {
                device: input[0].device(),
                dtype: Some(input[0].dtype()),
            },
        );
        for (input, noise_level) in input.into_iter().zip(noise_levels.into_iter()).rev() {
            a = self.u_net.forward(input, noise_level, a);
        }
        a
    }
}

fn tensor_pyramid<B: Backend>(sizes: impl Iterator<Item = [usize; 2]>, tensor: Tensor<B, 4>) -> Vec<Tensor<B, 4>> {
    let [_, _, height, width] = tensor.dims();

    sizes
        .map(|size| {
            if size == [height, width] {
                tensor.clone()
            } else {
                adaptive_avg_pool2d(tensor.clone(), size)
            }
        })
        .collect()
}
