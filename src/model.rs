use crate::model::u_net::{UNet, UNetConfig};
use burn::config::Config;
use burn::module::Module;
use burn::tensor::backend::Backend;
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
}

impl DiffusionModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DiffusionModel<B> {
        let u_net = self.u_net.init(device);
        DiffusionModel { u_net }
    }
}

impl<B: Backend> DiffusionModel<B> {
    pub fn input_sizes(&self, base_size: [usize; 2]) -> impl Iterator<Item = [usize; 2]> {
        assert_eq!(base_size[0].next_power_of_two(), base_size[0]);
        assert_eq!(base_size[1].next_power_of_two(), base_size[1]);
        let minimum_input_size = self.u_net.minimum_input_size();
        iter::successors(Some(base_size), |&size| Some(self.u_net.expected_insert_size(size)))
            .take_while(move |&size| minimum_input_size[0] <= size[0] && minimum_input_size[1] <= size[1])
    }

    pub fn forward(&self, input: Vec<Tensor<B, 4>>, _noise_level: Tensor<B, 4>) -> Tensor<B, 4> {
        assert!(
            input
                .iter()
                .map(Tensor::dims)
                .map(|[_, _, w, h]| [w, h])
                .eq(self.input_sizes([input[0].dims()[2], input[0].dims()[3]]))
        );
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
        for input in input.into_iter().rev() {
            a = self.u_net.forward(input, a);
        }
        a
    }
}
