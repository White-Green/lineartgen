"""A small full-resolution residual CNN for each recursive image-pyramid stage.

Only the external image pyramid resizes images. This stage never pools or
strides its feature maps. Width nesting is not implemented by this architecture.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from .output import denoised_output
from .unet import progress_basis
from .upsample import nearest_double


def local_contrast_cue(guide):
    """A bounded local shape cue, without changing the original guide or anchor."""
    ink = jnp.clip((1 - guide) / 2, 0, 1)
    peak = jax.lax.reduce_window(ink, -jnp.inf, jax.lax.max, (1, 33, 33, 1), (1, 1, 1, 1), "SAME")
    return 1 - 0.8 * ink / jnp.maximum(peak, 0.02)


class LocalBlock(nnx.Module):
    def __init__(self, width, *, rngs, pyramid_conditioning=False, dilation=1):
        self.first = nnx.Conv(
            width, width, (3, 3), padding="SAME", kernel_dilation=dilation, rngs=rngs
        )
        self.second = nnx.Conv(
            width, width, (3, 3), padding="SAME", kernel_dilation=dilation, rngs=rngs
        )
        self.modulation = nnx.Conv(
            9,
            2 * width,
            (1, 1),
            kernel_init=nnx.initializers.zeros_init(),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )
        if pyramid_conditioning:
            self.pyramid_modulation = nnx.Conv(
                1,
                2 * width,
                (1, 1),
                use_bias=False,
                kernel_init=nnx.initializers.zeros_init(),
                rngs=rngs,
            )

    def __call__(self, x, basis, pyramid_code=None):
        modulation = self.modulation(basis)
        if pyramid_code is not None:
            modulation = modulation + self.pyramid_modulation(pyramid_code)
        gain, shift = jnp.split(modulation, 2, axis=-1)
        h = nnx.silu(self.first(x) * (1 + gain) + shift)
        return nnx.silu(x + self.second(h))


class LocalCNN(nnx.Module):
    def __init__(self, config, *, rngs):
        if config.architecture != "local":
            raise ValueError("LocalCNN requires the local architecture")
        self.source_conditioning = True
        self.progress_conditioning = True
        self.progress_modulation = "fourier_film"
        self.output_mode = "residual"
        self.output_clip_gradient = "hard"
        self.sampling_mode = config.sampling_mode
        self.residual_base = config.residual_base
        self.feature_input = config.feature_input
        self.source_cue = config.source_cue
        self.source_preprocess = config.source_preprocess
        self.pyramid_conditioning = config.pyramid_conditioning
        self.local_dilations = config.local_dilations or (1,) * config.local_blocks
        width = config.channels[0]
        self.stem = nnx.Conv(4, width, (3, 3), padding="SAME", rngs=rngs)
        self.stem_modulation = nnx.Conv(
            9,
            2 * width,
            (1, 1),
            kernel_init=nnx.initializers.zeros_init(),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )
        if self.pyramid_conditioning != "none":
            self.stem_pyramid_modulation = nnx.Conv(
                1,
                2 * width,
                (1, 1),
                use_bias=False,
                kernel_init=nnx.initializers.zeros_init(),
                rngs=rngs,
            )
        self.blocks = nnx.List(
            [
                LocalBlock(
                    width,
                    rngs=rngs,
                    pyramid_conditioning=self.pyramid_conditioning != "none",
                    dilation=dilation,
                )
                for dilation in self.local_dilations
            ]
        )
        self.head = nnx.Conv(
            width,
            1,
            (1, 1),
            kernel_init=nnx.initializers.zeros_init(),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )

    def __call__(
        self,
        image,
        noise_level,
        insert,
        guide=None,
        progress=None,
        *,
        pyramid_level=0,
        return_preclip=False,
        return_network_output=False,
    ):
        if (
            guide is None
            or guide.shape != image.shape
            or progress is None
            or progress.shape != image.shape
        ):
            raise ValueError("LocalCNN requires matching source and progress maps")
        if insert.shape != (image.shape[0], image.shape[1] // 2, image.shape[2] // 2, 1):
            raise ValueError("Insert must be the immediate smaller image")
        if image.shape[1] % 2 or image.shape[2] % 2:
            raise ValueError("LocalCNN requires even image dimensions")
        up = nearest_double(insert)
        basis = progress_basis(progress)
        pyramid_code = None
        if self.pyramid_conditioning != "none":
            value = 1 + pyramid_level / 5 if self.pyramid_conditioning == "level" else 1
            pyramid_code = jnp.full_like(image, value)
        signal = guide if self.feature_input == "source" else image
        if self.source_cue == "local_peak":
            signal = local_contrast_cue(guide)
        features = jnp.concatenate((signal - 1, guide - 1, noise_level, up - 1), axis=-1)
        modulation = self.stem_modulation(basis)
        if pyramid_code is not None:
            modulation = modulation + self.stem_pyramid_modulation(pyramid_code)
        gain, shift = jnp.split(modulation, 2, axis=-1)
        x = nnx.silu(self.stem(features) * (1 + gain) + shift)
        for block in self.blocks:
            x = block(x, basis, pyramid_code)
        output = self.head(x)
        if return_network_output:
            if return_preclip:
                raise ValueError("Raw output and preclip modes are mutually exclusive")
            return output
        anchor = (
            jnp.where(noise_level > 0, guide, image) if self.residual_base == "source" else image
        )
        return denoised_output(
            output, anchor, noise_level, "residual", "hard", return_preclip=return_preclip
        )
