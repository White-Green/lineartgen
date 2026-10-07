"""One resolution of the recursive U-Net. All arrays use [batch, height, width, channels].

The image and noise map enter the encoder together. A smaller prediction is
inserted after the first pool. Skips default to after pooling. Optional pre-pool
skips retain detail above the unchanged pooled bottleneck. The final head also
sees the original image.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from lineartgen.config import ModelConfig

from .blocks import ConvBlock
from .output import denoised_output


class UNet(nnx.Module):
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        self.minimum_size = config.minimum_size
        self.output_mode = config.output_mode
        self.output_clip_gradient = config.output_clip_gradient
        self.skip_mode = config.skip_mode
        self.bottleneck_dilation = config.bottleneck_dilation
        encoder_inputs = [2] + [
            channels + (1 if index == 0 else 0)
            for index, channels in enumerate(config.channels[:-1])
        ]
        encoder_outputs = [config.channels[0] + 1, *config.channels[1:]]

        def block(inputs, outputs, dilation=1):
            return ConvBlock(
                inputs,
                outputs,
                kernel_size=config.kernel_size,
                depth=config.convolutions_per_block,
                dilation=dilation,
                rngs=rngs,
            )

        self.encoders = nnx.List(
            [
                block(
                    inputs,
                    outputs,
                    config.bottleneck_dilation if index == len(config.channels) - 1 else 1,
                )
                for index, (inputs, outputs) in enumerate(
                    zip(encoder_inputs, config.channels, strict=True)
                )
            ]
        )
        decoders = []
        previous = 0
        for index, (skip_channels, outputs) in enumerate(
            zip(reversed(encoder_outputs), reversed(encoder_inputs), strict=True)
        ):
            decoders.append(
                block(
                    skip_channels + previous,
                    outputs,
                    config.bottleneck_dilation if index == 0 else 1,
                )
            )
            previous = outputs
        self.decoders = nnx.List(decoders)
        self.head = block(1 + previous, 1)

    def __call__(self, image, noise_level, insert):
        def upsample(x):
            return jax.image.resize(
                x, (x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3]), "nearest"
            )

        x = jnp.concatenate((image, noise_level), axis=-1)
        skips = []
        for index, encoder in enumerate(self.encoders):
            features = encoder(x)
            x = nnx.max_pool(features, window_shape=(2, 2), strides=(2, 2), padding="VALID")
            if index == 0:
                x = jnp.concatenate((x, insert), axis=-1)
            if self.skip_mode == "pre_pool" and index < len(self.encoders) - 1:
                skips.append(
                    jnp.concatenate((features, upsample(insert)), -1) if index == 0 else features
                )
            else:
                skips.append(x)
        for index, (decoder, skip) in enumerate(zip(self.decoders, reversed(skips), strict=True)):
            if self.skip_mode == "pre_pool" and index > 0:
                x = jnp.concatenate((skip, upsample(x)), axis=-1)
            else:
                x = upsample(skip if index == 0 else jnp.concatenate((skip, x), axis=-1))
            x = decoder(x)
        return denoised_output(
            self.head(jnp.concatenate((image, x), -1), linear_output=True),
            image,
            noise_level,
            self.output_mode,
            self.output_clip_gradient,
        )
