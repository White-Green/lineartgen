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
from .upsample import nearest_double


def progress_basis(progress):
    """Nine bounded time features; no teacher or pooled image information."""
    tau = 1 - progress
    angles = jnp.pi * tau * jnp.asarray([1, 2, 4, 8], dtype=progress.dtype)
    return jnp.concatenate((tau, jnp.sin(angles), jnp.cos(angles)), axis=-1)


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
        self.residual_base = config.residual_base
        self.feature_input = config.feature_input
        self.sampling_mode = config.sampling_mode
        self.source_conditioning = config.source_conditioning
        self.progress_conditioning = config.progress_conditioning
        self.progress_modulation = config.progress_modulation
        self.guide_encoding = config.guide_encoding
        self.detail_channels = config.detail_channels
        self.skip_mode = config.skip_mode
        self.bottleneck_dilation = config.bottleneck_dilation
        encoder_inputs = [2] + [
            channels + (1 if index == 0 else 0)
            for index, channels in enumerate(config.channels[:-1])
        ]
        encoder_outputs = [config.channels[0] + 1, *config.channels[1:]]

        def block(inputs, outputs, dilation=1, hidden_channels=None):
            return ConvBlock(
                inputs,
                outputs,
                kernel_size=config.kernel_size,
                depth=config.convolutions_per_block,
                dilation=dilation,
                hidden_channels=hidden_channels,
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
            if index == len(config.channels) - 1:
                outputs = config.final_decoder_channels
            decoders.append(
                block(
                    skip_channels + previous,
                    outputs,
                    config.bottleneck_dilation if index == 0 else 1,
                )
            )
            previous = outputs
        self.decoders = nnx.List(decoders)
        self.head = block(1 + previous, 1, hidden_channels=config.head_hidden_channels)
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

        if self.progress_conditioning:
            # Construct last: every inherited parameter and random draw is unchanged.
            def progress_projection(channels):
                return nnx.Conv(
                    1,
                    channels,
                    (1, 1),
                    padding="SAME",
                    use_bias=False,
                    kernel_init=nnx.initializers.zeros,
                    rngs=rngs,
                )

            self.progress_encoder = progress_projection(self.encoders[0].layers[0].out_features)
            self.progress_head = progress_projection(self.head.layers[0].out_features)
            if self.detail_channels:
                self.progress_detail = progress_projection(self.detail_channels)

        if self.progress_modulation == "fourier_film":
            # Append after all inherited draws. Zero projections preserve the
            # old function initially while enabling time-dependent gain and bias.
            blocks = [*self.encoders, *self.decoders, self.head]
            if self.detail_channels:
                blocks.append(self.detail_head)
            self.progress_modulators = nnx.List(
                [
                    nnx.Conv(
                        9,
                        2 * block.layers[0].out_features,
                        (1, 1),
                        padding="SAME",
                        kernel_init=nnx.initializers.zeros,
                        bias_init=nnx.initializers.zeros,
                        rngs=rngs,
                    )
                    for block in blocks
                ]
            )

    def __call__(
        self,
        image,
        noise_level,
        insert,
        guide=None,
        progress=None,
        *,
        return_preclip=False,
        return_features=False,
        return_network_output=False,
    ):
        sampled = image
        if self.source_conditioning:
            if guide is None or guide.shape != image.shape:
                raise ValueError("Source-conditioned UNet requires a matching original rough guide")
            encoded = encode_guide(guide, self.guide_encoding)
            encoder_bias, head_bias = self.guide_encoder(encoded), self.guide_head(encoded)
        else:
            if guide is not None:
                raise ValueError("This UNet is not configured for source conditioning")
            encoder_bias = head_bias = None
        detail_bias = None
        if self.progress_conditioning:
            if progress is None or progress.shape != image.shape:
                raise ValueError("Progress-conditioned UNet requires a matching progress map")
            feature = 1 - progress
            encoder_bias = encoder_bias + self.progress_encoder(feature)
            head_bias = head_bias + self.progress_head(feature)
            if self.detail_channels:
                detail_bias = self.progress_detail(feature)
        elif progress is not None:
            raise ValueError("This UNet is not configured for progress conditioning")

        if self.feature_input == "source":
            image = guide

        basis = progress_basis(progress) if self.progress_modulation == "fourier_film" else None

        def modulation(index, x):
            if basis is None:
                return None
            features = jax.image.resize(basis, (*x.shape[:3], basis.shape[-1]), "nearest")
            return jnp.split(self.progress_modulators[index](features), 2, axis=-1)

        upsample = nearest_double

        x = jnp.concatenate((image, noise_level), axis=-1)
        skips = []
        for index, encoder in enumerate(self.encoders):
            features = encoder(
                x,
                first_bias=encoder_bias if index == 0 else None,
                first_modulation=modulation(index, x),
            )
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
            x = decoder(x, first_modulation=modulation(len(self.encoders) + index, x))
        output = self.head(
            jnp.concatenate((image, x), -1),
            linear_output=True,
            first_bias=head_bias,
            first_modulation=modulation(len(self.encoders) + len(self.decoders), x),
        )
        if self.detail_channels:
            output = output + self.detail_head(
                jnp.concatenate((image, noise_level, encoded, full_resolution, x), -1),
                linear_output=True,
                first_bias=detail_bias,
                first_modulation=modulation(len(self.encoders) + len(self.decoders) + 1, x),
            )
        if return_network_output:
            if return_preclip or return_features:
                raise ValueError("Raw network output cannot be combined with other return modes")
            return output
        # A source anchor asks the head to restore the original rough image,
        # without also having to cancel every sampled Gaussian pixel. Keep
        # the incoming image at protected pixels, including repeated sampling.
        anchor = (
            jnp.where(noise_level > 0, guide, sampled) if self.residual_base == "source" else image
        )
        result = denoised_output(
            output,
            anchor,
            noise_level,
            self.output_mode,
            self.output_clip_gradient,
            return_preclip=return_preclip,
            scale_floor=self.residual_scale_floor,
        )
        if return_features:
            if not self.detail_channels or not self.source_conditioning:
                raise ValueError("Auxiliary features require source conditioning and detail head")
            return result, jnp.concatenate((image, noise_level, encoded, full_resolution, x), -1)
        return result
