use burn::config::Config;
use burn::module::Module;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig};
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::tensor::Tensor;
use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UNetStepConfig {
    encoder_kernel_size: usize,
    decoder_kernel_size: usize,
    pool_size: usize,
    down_channels_mul: f64,
    transposed_channels_mul: f64,
}

impl Default for UNetStepConfig {
    fn default() -> Self {
        UNetStepConfig {
            encoder_kernel_size: 3,
            decoder_kernel_size: 3,
            pool_size: 2,
            down_channels_mul: 2.0,
            transposed_channels_mul: 1.0,
        }
    }
}

#[derive(Config, Debug)]
pub struct UNetConfig {
    #[config(default = "vec![UNetStepConfig::default(); 7]")]
    pub sizes: Vec<UNetStepConfig>,
    #[config(default = "1")]
    pub insert_channels: usize,
    #[config(default = "2")]
    pub input_channels: usize,
    #[config(default = "1")]
    pub output_channels: usize,
    #[config(default = "3")]
    pub final_kernel_size: usize,
}

#[derive(Module, Debug)]
struct UNetEncoder<B: Backend> {
    conv: Conv2d<B>,
    pool: MaxPool2d,
    input_channels: usize,
    output_channels: usize,
    shrink_rate: usize,
}

#[derive(Module, Debug)]
pub struct UNet<B: Backend> {
    encoders: Vec<UNetEncoder<B>>,
    decoders: Vec<ConvTranspose2d<B>>,
    final_conv: Conv2d<B>,
}

impl UNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> UNet<B> {
        let encoders = self
            .sizes
            .iter()
            .scan(
                (self.input_channels, 1),
                |(channels, size),
                 &UNetStepConfig {
                     encoder_kernel_size,
                     pool_size,
                     down_channels_mul,
                     ..
                 }| {
                    let input_channels = *channels;
                    *channels = (*channels as f64 * down_channels_mul).round() as usize;
                    let output_channels = *channels;
                    let conv = Conv2dConfig::new([input_channels, output_channels], [encoder_kernel_size; 2])
                        .with_padding(PaddingConfig2d::Same)
                        .init(device);
                    let pool = MaxPool2dConfig::new([pool_size; 2]).init();
                    *size *= pool_size;
                    Some(UNetEncoder {
                        conv,
                        pool,
                        input_channels,
                        output_channels,
                        shrink_rate: *size,
                    })
                },
            )
            .collect::<Vec<_>>();
        let decoders = self
            .sizes
            .iter()
            .enumerate()
            .rev()
            .zip(encoders.iter().rev())
            .scan(
                0,
                |channels,
                 (
                    (
                        depth,
                        &UNetStepConfig {
                            decoder_kernel_size,
                            pool_size,
                            transposed_channels_mul,
                            ..
                        },
                    ),
                    encoder,
                )| {
                    let input_channels = if depth == 0 {
                        *channels + self.insert_channels + encoder.output_channels
                    } else {
                        *channels + encoder.output_channels
                    };
                    let output_channels = (encoder.input_channels as f64 * transposed_channels_mul).round() as usize;
                    let padding = decoder_kernel_size.saturating_sub(pool_size).div_ceil(2);
                    let padding_out = pool_size + 2 * padding - decoder_kernel_size;
                    *channels = output_channels;
                    Some(
                        ConvTranspose2dConfig::new([input_channels, output_channels], [decoder_kernel_size; 2])
                            .with_stride([pool_size; 2])
                            .with_padding([padding; 2])
                            .with_padding_out([padding_out; 2])
                            .init(device),
                    )
                },
            )
            .collect();
        let decoder_output_channels =
            (self.input_channels as f64 * self.sizes[0].transposed_channels_mul).round() as usize;
        let final_conv = Conv2dConfig::new(
            [self.input_channels + decoder_output_channels, self.output_channels],
            [self.final_kernel_size; 2],
        )
        .with_padding(PaddingConfig2d::Same)
        .init(device);

        UNet {
            encoders,
            decoders,
            final_conv,
        }
    }
}

impl<B: Backend> UNet<B> {
    pub fn minimum_input_size(&self) -> [usize; 2] {
        let shrink_rate = self.encoders.last().map(|encoder| encoder.shrink_rate).unwrap_or(1);
        [shrink_rate; 2]
    }

    pub fn expected_insert_size(&self, input_size: [usize; 2]) -> [usize; 2] {
        let shrink_rate = self.encoders.first().map(|encoder| encoder.shrink_rate).unwrap_or(1);

        assert_eq!(
            input_size[0] % shrink_rate,
            0,
            "input height must be divisible by the first encoder shrink rate"
        );
        assert_eq!(
            input_size[1] % shrink_rate,
            0,
            "input width must be divisible by the first encoder shrink rate"
        );

        [input_size[0] / shrink_rate, input_size[1] / shrink_rate]
    }

    pub fn forward(&self, input: Tensor<B, 4>, noise_level: Tensor<B, 4>, insert: Tensor<B, 4>) -> Tensor<B, 4> {
        assert_eq!(input.dims()[2].next_power_of_two(), input.dims()[2]);
        assert_eq!(input.dims()[3].next_power_of_two(), input.dims()[3]);
        assert_eq!(input.dims()[1], 1);
        assert_eq!(noise_level.dims(), input.dims());
        let conditioned_input = Tensor::cat(vec![input, noise_level], 1);
        let mut x = conditioned_input.clone();
        let mut skips = Vec::with_capacity(self.encoders.len());

        for encoder in self.encoders.iter() {
            assert_eq!(x.dims()[1], encoder.input_channels);
            let c = relu(encoder.conv.forward(x));
            x = encoder.pool.forward(c);
            skips.push(x.clone());
            assert_eq!(x.dims()[1], encoder.output_channels);
            assert_eq!(x.dims()[2] * encoder.shrink_rate, conditioned_input.dims()[2]);
            assert_eq!(x.dims()[3] * encoder.shrink_rate, conditioned_input.dims()[3]);
        }

        for (index, decoder) in self.decoders.iter().enumerate() {
            let skip = skips.pop().expect("UNet decoder count should match encoder count");
            let is_last = index + 1 == self.decoders.len();

            let mut stacks = Vec::new();
            stacks.push(skip);
            if is_last {
                stacks.push(insert.clone());
            }
            if index != 0 {
                stacks.push(x);
            }

            let cat = Tensor::cat(stacks, 1);
            x = relu(decoder.forward(cat));
        }

        let result = Tensor::clamp(
            self.final_conv.forward(Tensor::cat(vec![conditioned_input, x], 1)),
            -1.0,
            1.0,
        );
        assert_eq!(result.dims()[1], self.final_conv.weight.dims()[0]);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_unet_forwards_single_1024_image() {
        let device = Default::default();
        let model = UNetConfig::new().init::<burn::backend::Flex>(&device);
        let input = Tensor::<_, 4>::zeros([1, 1, 1024, 1024], &device);
        let noise_level = Tensor::<_, 4>::zeros([1, 1, 1024, 1024], &device);
        let [insert_height, insert_width] = model.expected_insert_size([1024, 1024]);
        let insert = Tensor::<_, 4>::zeros([1, 1, insert_height, insert_width], &device);

        let output = model.forward(input, noise_level, insert);

        assert_eq!(output.dims(), [1, 1, 1024, 1024]);
    }

    #[test]
    fn default_unet_reports_input_and_insert_sizes() {
        let device = Default::default();
        let model = UNetConfig::new().init::<burn::backend::Flex>(&device);

        assert_eq!(model.minimum_input_size(), [128, 128]);
        assert_eq!(model.expected_insert_size([1024, 1024]), [512, 512]);
    }
}
