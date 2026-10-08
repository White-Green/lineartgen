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


def encode_guide(guide, encoding):
    """Encode the downsampled original rough source, never a clean teacher.

    Ink encodings are zero on white even at padding boundaries. The square-root
    transform boosts faint ink without adding parameters or discarding location.
    """
    if encoding == "raw":
        return guide
    ink = (1 - guide) * 0.5
    if encoding == "ink":
        return ink
    if encoding == "sqrt_ink":
        return jnp.sqrt(jnp.clip(ink, 0, 1))
    raise ValueError(f"Unknown guide encoding: {encoding}")


class UNet(nnx.Module):
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        self.minimum_size = config.minimum_size
        self.output_mode = config.output_mode
        self.output_clip_gradient = config.output_clip_gradient
        self.residual_scale_floor = config.residual_scale_floor
        self.source_conditioning = config.source_conditioning
        self.guide_encoding = config.guide_encoding
        self.detail_channels = config.detail_channels
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
        if self.source_conditioning:
            # Zero additions preserve the existing model at initialization. The
            # original rough image is available at inference, unlike the teacher.
            def guide_projection(channels):
                return nnx.Conv(
                    1,
                    channels,
                    (config.kernel_size, config.kernel_size),
                    padding="SAME",
                    use_bias=False,
                    kernel_init=nnx.initializers.zeros,
                    rngs=rngs,
                )

            self.guide_encoder = guide_projection(self.encoders[0].layers[0].out_features)
            self.guide_head = guide_projection(self.head.layers[0].out_features)
        if self.detail_channels:
            # Construct after every inherited layer so its initialization is unchanged.
            # The final zero projection preserves the original function, while the
            # randomly initialized hidden layers let that projection learn immediately.
            self.detail_head = ConvBlock(
                3 + config.channels[0] + previous,
                1,
                kernel_size=config.kernel_size,
                depth=3,
                hidden_channels=self.detail_channels,
                rngs=rngs,
            )
            self.detail_head.layers[-1].kernel[...] = 0
            self.detail_head.layers[-1].bias[...] = 0

    def __call__(self, image, noise_level, insert, guide=None, *, return_preclip=False):
        if self.source_conditioning:
            if guide is None or guide.shape != image.shape:
                raise ValueError("Source-conditioned UNet requires a matching original rough guide")
            encoded = encode_guide(guide, self.guide_encoding)
            encoder_bias, head_bias = self.guide_encoder(encoded), self.guide_head(encoded)
        else:
            if guide is not None:
                raise ValueError("This UNet is not configured for source conditioning")
            encoder_bias = head_bias = None

        def upsample(x):
            return jax.image.resize(
                x, (x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3]), "nearest"
            )

        x = jnp.concatenate((image, noise_level), axis=-1)
        skips = []
        for index, encoder in enumerate(self.encoders):
            features = encoder(x, first_bias=encoder_bias if index == 0 else None)
            if self.detail_channels and index == 0:
                full_resolution = features
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
        output = self.head(
            jnp.concatenate((image, x), -1), linear_output=True, first_bias=head_bias
        )
        if self.detail_channels:
            output = output + self.detail_head(
                jnp.concatenate((image, noise_level, encoded, full_resolution, x), -1),
                linear_output=True,
            )
        return denoised_output(
            output,
            image,
            noise_level,
            self.output_mode,
            self.output_clip_gradient,
            return_preclip=return_preclip,
            scale_floor=self.residual_scale_floor,
        )
