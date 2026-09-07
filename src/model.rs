use crate::model::u_net::{UNet, UNetConfig};
use burn::config::Config;
use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorCreationOptions};

pub mod u_net;

#[derive(Config, Debug)]
pub struct DiffusionModelConfig {
    #[config(default = "UNetConfig::new()")]
    pub u_net: UNetConfig,
}

#[derive(Module, Debug)]
pub struct DiffusionModel<B: Backend> {
    u_net: UNet<B>,
}

impl DiffusionModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DiffusionModel<B> {
        DiffusionModel {
            u_net: self.u_net.init(device),
        }
    }
}

impl<B: Backend> DiffusionModel<B> {
    pub fn minimum_input_size(&self) -> [usize; 2] {
        self.u_net.minimum_input_size()
    }

    pub fn expected_insert_size(&self, input_size: [usize; 2]) -> [usize; 2] {
        self.u_net.expected_insert_size(input_size)
    }

    pub fn insert_channels(&self) -> usize {
        self.u_net.insert_channels()
    }

    /// Processes exactly one resolution. Recursive orchestration belongs to the caller.
    pub fn forward(
        &self,
        input: Tensor<B, 4>,
        noise_level: Tensor<B, 4>,
        insert: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 4> {
        let [batch_size, _, height, width] = input.dims();
        let [insert_height, insert_width] = self.expected_insert_size([height, width]);
        let expected_insert_dims = [batch_size, self.insert_channels(), insert_height, insert_width];
        let insert = match insert {
            Some(insert) => {
                assert_eq!(
                    insert.dims(),
                    expected_insert_dims,
                    "insert shape must match the first encoder output"
                );
                insert
            }
            None => Tensor::full(
                expected_insert_dims,
                0.0,
                TensorCreationOptions {
                    device: input.device(),
                    dtype: Some(input.dtype()),
                },
            ),
        };

        self.u_net.forward(input, noise_level, insert)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_step_forward_accepts_an_insert() {
        let device = Default::default();
        let model = DiffusionModelConfig::new().init::<burn::backend::Flex>(&device);
        let input = Tensor::<_, 4>::zeros([1, 1, 128, 128], &device);
        let noise_level = Tensor::<_, 4>::zeros([1, 1, 128, 128], &device);
        let [insert_height, insert_width] = model.expected_insert_size([128, 128]);
        let insert = Tensor::<_, 4>::zeros([1, model.insert_channels(), insert_height, insert_width], &device);

        let output = model.forward(input, noise_level, Some(insert));

        assert_eq!(output.dims(), [1, 1, 128, 128]);
    }

    #[test]
    fn missing_insert_is_equivalent_to_an_explicit_zero_insert() {
        let device = Default::default();
        let model = DiffusionModelConfig::new().init::<burn::backend::Flex>(&device);
        let input = Tensor::<_, 4>::zeros([1, 1, 32, 32], &device);
        let noise_level = Tensor::<_, 4>::zeros([1, 1, 32, 32], &device);
        let [insert_height, insert_width] = model.expected_insert_size([32, 32]);
        let insert = Tensor::<_, 4>::zeros([1, model.insert_channels(), insert_height, insert_width], &device);

        let implicit = model.forward(input.clone(), noise_level.clone(), None);
        let explicit = model.forward(input, noise_level, Some(insert));

        assert_eq!(
            implicit.into_data().into_vec::<f32>().unwrap(),
            explicit.into_data().into_vec::<f32>().unwrap()
        );
    }

    #[test]
    #[should_panic(expected = "insert shape must match the first encoder output")]
    fn rejects_an_insert_with_the_wrong_shape() {
        let device = Default::default();
        let model = DiffusionModelConfig::new().init::<burn::backend::Flex>(&device);
        let input = Tensor::<_, 4>::zeros([1, 1, 32, 32], &device);
        let noise_level = Tensor::<_, 4>::zeros([1, 1, 32, 32], &device);
        let insert = Tensor::<_, 4>::zeros([1, 1, 8, 8], &device);

        let _ = model.forward(input, noise_level, Some(insert));
    }
}
