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

#[derive(Config, Debug)]
pub struct LineartLossConfig {
    /// 多段avg poolで黒密度を比較
    // #[config(default = "0.25")]
    #[config(default = "0.0")]
    pub density_weight: f64,
    /// Sobelカーネルの畳み込み結果を比較
    #[config(default = "0.0")]
    // #[config(default = "0.0")]
    pub edge_weight: f64,
    /// 正解線の近傍外に出た黒を罰する
    // #[config(default = "0.10")]
    #[config(default = "0.0")]
    pub speckle_weight: f64,
    /// 灰色っぽい中間値を罰する
    // #[config(default = "0.05")]
    #[config(default = "0.05")]
    pub contrast_weight: f64,
    /// 正解を参照せず、黒画素が近傍の黒supportを持つようにする
    #[config(default = "0.1")]
    pub support_weight: f64,
    /// 正解を参照せず、黒画素が線状の方向supportを持つようにする
    #[config(default = "0.1")]
    pub direction_weight: f64,
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
    #[module(skip)]
    micro_batch_size: usize,
    #[module(skip)]
    noise_pool_size: Option<usize>,
    #[module(skip)]
    lineart_loss: LineartLossConfig,
}

impl DiffusionModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DiffusionModel<B> {
        let u_net = self.u_net.init(device);
        DiffusionModel {
            u_net,
            sample_noise_levels: Vec::new(),
            sample_denoising_steps: 0,
            balance_loss_by_tone: false,
            micro_batch_size: 32,
            noise_pool_size: None,
            lineart_loss: LineartLossConfig::new(),
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

    pub fn with_training_batching(mut self, micro_batch_size: usize, noise_pool_size: Option<usize>) -> Self {
        self.micro_batch_size = micro_batch_size.max(1);
        self.noise_pool_size = noise_pool_size;
        self
    }

    pub fn with_lineart_loss_config(mut self, lineart_loss: LineartLossConfig) -> Self {
        self.lineart_loss = lineart_loss;
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

    pub fn micro_batch_size(&self) -> usize {
        self.micro_batch_size.max(1)
    }

    pub fn noise_pool_size(&self, batch_size: usize) -> usize {
        self.noise_pool_size.unwrap_or(batch_size).max(batch_size)
    }

    pub fn lineart_loss_config(&self) -> &LineartLossConfig {
        &self.lineart_loss
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
            0.0,
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

    pub fn forward_teacher_forced(
        &self,
        input: Vec<Tensor<B, 4>>,
        noise_level: Tensor<B, 4>,
        clean: Vec<Tensor<B, 4>>,
    ) -> Vec<Tensor<B, 4>> {
        assert_eq!(input.len(), clean.len());
        assert!(
            input
                .iter()
                .map(Tensor::dims)
                .map(|[_, _, w, h]| [w, h])
                .eq(self.input_sizes([input[0].dims()[2], input[0].dims()[3]]))
        );
        assert!(
            input
                .iter()
                .zip(clean.iter())
                .all(|(input, clean)| input.dims() == clean.dims())
        );

        let noise_levels = tensor_pyramid(self.input_sizes([input[0].dims()[2], input[0].dims()[3]]), noise_level);
        let mut outputs = Vec::with_capacity(input.len());

        for index in 0..input.len() {
            let insert = match clean.get(index + 1) {
                Some(clean) => clean.clone(),
                None => {
                    let [batches, _, height, width] = input[index].dims();
                    let [empty_h, empty_w] = self.u_net.expected_insert_size([height, width]);
                    Tensor::full(
                        [batches, 1, empty_h, empty_w],
                        0.0,
                        TensorCreationOptions {
                            device: input[index].device(),
                            dtype: Some(input[index].dtype()),
                        },
                    )
                }
            };
            outputs.push(
                self.u_net
                    .forward(input[index].clone(), noise_levels[index].clone(), insert),
            );
        }

        outputs
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Tensor;

    #[test]
    fn teacher_forced_forward_returns_all_scales_for_default_model() {
        let device = Default::default();
        let model = DiffusionModelConfig::new().init::<burn::backend::Flex>(&device);
        let input = Tensor::<_, 4>::zeros([1, 1, 1024, 1024], &device);
        let noise_level = Tensor::<_, 4>::zeros([1, 1, 1024, 1024], &device);
        let inputs = tensor_pyramid(model.input_sizes([1024, 1024]), input.clone());
        let clean = tensor_pyramid(model.input_sizes([1024, 1024]), input);

        let outputs = model.forward_teacher_forced(inputs, noise_level, clean);

        let expected_sizes = model.input_sizes([1024, 1024]).collect::<Vec<_>>();
        assert_eq!(outputs.len(), expected_sizes.len());
        for (output, [height, width]) in outputs.iter().zip(expected_sizes) {
            assert_eq!(output.dims(), [1, 1, height, width]);
        }
    }
}
