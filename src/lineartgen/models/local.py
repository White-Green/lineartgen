"""A small full-resolution residual CNN for each recursive image-pyramid stage.

Only the external image pyramid resizes images. This stage never pools or
strides its feature maps. Width nesting is not implemented by this architecture.
"""

import jax.numpy as jnp
from flax import nnx

from .output import denoised_output
from .unet import progress_basis
from .upsample import nearest_double


class LocalBlock(nnx.Module):
    def __init__(self, width, *, rngs):
        self.first = nnx.Conv(width, width, (3, 3), padding="SAME", rngs=rngs)
        self.second = nnx.Conv(width, width, (3, 3), padding="SAME", rngs=rngs)
        self.modulation = nnx.Conv(
            9,
            2 * width,
            (1, 1),
            kernel_init=nnx.initializers.zeros_init(),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )

    def __call__(self, x, basis):
        gain, shift = jnp.split(self.modulation(basis), 2, axis=-1)
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
        self.sampling_mode = "diffusion"
        self.residual_base = "noisy"
        self.feature_input = "noisy"
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
        self.blocks = nnx.List([LocalBlock(width, rngs=rngs) for _ in range(config.local_blocks)])
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
        features = jnp.concatenate((image - 1, guide - 1, noise_level, up - 1), axis=-1)
        gain, shift = jnp.split(self.stem_modulation(basis), 2, axis=-1)
        x = nnx.silu(self.stem(features) * (1 + gain) + shift)
        for block in self.blocks:
            x = block(x, basis)
        output = self.head(x)
        if return_network_output:
            if return_preclip:
                raise ValueError("Raw output and preclip modes are mutually exclusive")
            return output
        return denoised_output(
            output, image, noise_level, "residual", "hard", return_preclip=return_preclip
        )
