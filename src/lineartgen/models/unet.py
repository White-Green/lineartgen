"""One resolution of the recursive U-Net. All arrays use [batch, height, width, channels].

The image and noise map enter the encoder together. A smaller prediction is
inserted after the first pool. Skip tensors are saved after pooling, matching
the original architecture. The final head also sees the original image.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from lineartgen.config import ModelConfig

from .blocks import ConvBlock


class UNet(nnx.Module):
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        self.minimum_size = config.minimum_size
        encoder_inputs = [2] + [
            channels + (1 if index == 0 else 0)
            for index, channels in enumerate(config.channels[:-1])
        ]
        encoder_outputs = [config.channels[0] + 1, *config.channels[1:]]

        def block(inputs, outputs):
            return ConvBlock(
                inputs,
                outputs,
                kernel_size=config.kernel_size,
                depth=config.convolutions_per_block,
                rngs=rngs,
            )

        self.encoders = nnx.List(
            [
                block(inputs, outputs)
                for inputs, outputs in zip(encoder_inputs, config.channels, strict=True)
            ]
        )
        decoders = []
        previous = 0
        for skip_channels, outputs in zip(
            reversed(encoder_outputs), reversed(encoder_inputs), strict=True
        ):
            decoders.append(block(skip_channels + previous, outputs))
            previous = outputs
        self.decoders = nnx.List(decoders)
        self.head = block(1 + previous, 1)

    def __call__(self, image, noise_level, insert):
        x = jnp.concatenate((image, noise_level), axis=-1)
        skips = []
        for index, encoder in enumerate(self.encoders):
            x = nnx.max_pool(encoder(x), window_shape=(2, 2), strides=(2, 2), padding="VALID")
            if index == 0:
                x = jnp.concatenate((x, insert), axis=-1)
            skips.append(x)
        for index, (decoder, skip) in enumerate(zip(self.decoders, reversed(skips), strict=True)):
            x = skip if index == 0 else jnp.concatenate((skip, x), axis=-1)
            x = jax.image.resize(
                x, (x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3]), "nearest"
            )
            x = decoder(x)
        return jnp.clip(self.head(jnp.concatenate((image, x), -1), linear_output=True), -1, 1)
