use burn::config::Config;
use burn::module::Module;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig};
use burn::tensor::Tensor;
use burn::tensor::activation::{relu, tanh};
use burn::tensor::backend::Backend;

#[derive(Config, Debug)]
pub struct DiffusionModelConfig {
    #[config(default = "1")]
    pub input_channels: usize,
    #[config(default = "32")]
    pub base_channels: usize,
    #[config(default = "128")]
    pub latent_channels: usize,
}

#[derive(Module, Debug)]
pub struct DiffusionModel<B: Backend> {
    encoder1: Conv2d<B>,
    encoder2: Conv2d<B>,
    encoder3: Conv2d<B>,
    decoder1: ConvTranspose2d<B>,
    decoder2: ConvTranspose2d<B>,
    decoder3: ConvTranspose2d<B>,
}

impl DiffusionModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DiffusionModel<B> {
        let c = self.base_channels;
        let z = self.latent_channels;

        DiffusionModel {
            encoder1: downsample_conv(self.input_channels, c, device),
            encoder2: downsample_conv(c, c * 2, device),
            encoder3: downsample_conv(c * 2, z, device),
            decoder1: upsample_conv(z, c * 2, device),
            decoder2: upsample_conv(c * 2, c, device),
            decoder3: upsample_conv(c, self.input_channels, device),
        }
    }
}

impl<B: Backend> DiffusionModel<B> {
    pub fn forward(&self, input: Tensor<B, 4>, _noise_level: Tensor<B, 4>) -> Tensor<B, 4> {
        let encoded = self.encode(input);
        self.decode(encoded)
    }

    pub fn encode(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = relu(self.encoder1.forward(input));
        let x = relu(self.encoder2.forward(x));
        relu(self.encoder3.forward(x))
    }

    pub fn decode(&self, encoded: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = relu(self.decoder1.forward(encoded));
        let x = relu(self.decoder2.forward(x));
        tanh(self.decoder3.forward(x))
    }
}

fn downsample_conv<B: Backend>(channels_in: usize, channels_out: usize, device: &B::Device) -> Conv2d<B> {
    Conv2dConfig::new([channels_in, channels_out], [4, 4])
        .with_stride([2, 2])
        .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
        .init(device)
}

fn upsample_conv<B: Backend>(channels_in: usize, channels_out: usize, device: &B::Device) -> ConvTranspose2d<B> {
    ConvTranspose2dConfig::new([channels_in, channels_out], [4, 4])
        .with_stride([2, 2])
        .with_padding([1, 1])
        .init(device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BurnBackend;

    #[test]
    fn diffusion_model_preserves_image_shape() {
        let device = Default::default();
        let model = DiffusionModelConfig::new().init::<BurnBackend>(&device);
        let input = Tensor::<BurnBackend, 4>::zeros([2, 1, 128, 128], &device);
        let noise_level = Tensor::<BurnBackend, 4>::zeros([2, 1, 1, 1], &device);

        let output = model.forward(input, noise_level);

        assert_eq!(output.dims(), [2, 1, 128, 128]);
    }
}
