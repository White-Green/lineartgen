"""Research D4 group convolutions and an equivariant 4x4 subpixel field head.

Group coordinates are horizontal reflection followed by counterclockwise
quarter turns. Features transform by spatial action and left permutation.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from lineartgen.models.dilated import dilated_same
from lineartgen.models.unet import progress_basis
from lineartgen.models.upsample import nearest_double


def action(x, g, axes=(0, 1)):
    if g // 4:
        x = jnp.flip(x, axis=axes[1])
    return jnp.rot90(x, g % 4, axes=axes)


def _tables():
    grid = np.arange(9).reshape(3, 3)

    def move(x, g):
        return np.rot90(np.flip(x, 1) if g // 4 else x, g % 4)

    elements = [move(grid, g) for g in range(8)]
    table = tuple(
        tuple(
            next(k for k in range(8) if np.array_equal(move(elements[h], g), elements[k]))
            for h in range(8)
        )
        for g in range(8)
    )
    inverse = tuple(next(h for h in range(8) if table[g][h] == 0) for g in range(8))
    return table, inverse


PRODUCT, INVERSE = _tables()


def feature_action(x, g):
    """NHWGC regular representation, for tests and diagnostics."""
    return action(x, g, axes=(1, 2))[..., jnp.asarray(PRODUCT[INVERSE[g]]), :]


class GroupConv(nnx.Module):
    def __init__(self, inputs, outputs, *, lifting=False, dilation=1, rngs):
        self.lifting = lifting
        self.dilation = dilation
        self.width = outputs
        shape = (3, 3, inputs, outputs) if lifting else (3, 3, 8, inputs, outputs)
        self.kernel = nnx.Param(nnx.initializers.lecun_normal()(rngs.params(), shape))
        self.bias = nnx.Param(jnp.zeros(outputs))

    def expanded_kernel(self):
        base = self.kernel[...]
        pieces = []
        for g in range(8):
            kernel = base if self.lifting else base[:, :, jnp.asarray(PRODUCT[INVERSE[g]])]
            pieces.append(action(kernel, g).reshape(3, 3, -1, self.width))
        return jnp.concatenate(pieces, axis=-1)

    def __call__(self, x):
        kernel = self.expanded_kernel()
        if self.dilation == 1:
            y = jax.lax.conv_general_dilated(
                x, kernel, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
            )
        else:
            y = dilated_same(x, kernel, self.dilation)
        return y + jnp.tile(self.bias[...], 8)


class GroupFiLM(nnx.Module):
    def __init__(self, width, *, rngs):
        self.width = width
        self.linear = nnx.Conv(
            9,
            2 * width,
            (1, 1),
            kernel_init=nnx.initializers.zeros_init(),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )

    def __call__(self, x, basis):
        gain, shift = jnp.split(self.linear(basis), 2, axis=-1)
        y = x.reshape(*x.shape[:-1], 8, self.width)
        return (y * (1 + gain[..., None, :]) + shift[..., None, :]).reshape(x.shape)


class GroupBlock(nnx.Module):
    def __init__(self, width, dilation, *, rngs):
        self.first = GroupConv(width, width, dilation=dilation, rngs=rngs)
        self.second = GroupConv(width, width, dilation=dilation, rngs=rngs)
        self.modulation = GroupFiLM(width, rngs=rngs)

    def __call__(self, x, basis):
        y = nnx.silu(self.modulation(self.first(x), basis))
        return nnx.silu(x + self.second(y))


class GroupHead(nnx.Module):
    def __init__(self, width, *, rngs):
        self.linear = nnx.Linear(
            width,
            16,
            kernel_init=nnx.initializers.zeros_init(),
            bias_init=nnx.initializers.ones_init(),
            rngs=rngs,
        )

    def __call__(self, x):
        b, h, w, c = x.shape
        patches = self.linear(x.reshape(b, h, w, 8, c // 8)).reshape(b, h, w, 8, 4, 4)
        # Rotate each orientation's local subpixel coordinates into the image.
        parts = [action(patches[..., g, :, :], g, axes=(3, 4)) for g in range(8)]
        return jnp.stack(parts).mean(0).reshape(b, h, w, 16)


class GroupFieldCNN(nnx.Module):
    def __init__(self, config, *, rngs):
        assert config.feature_input == "source" and config.source_cue == "raw"
        assert config.pyramid_conditioning == "fourier"
        assert config.source_preprocess == "local_peak_full"
        self.research_model = "d4_group_local"
        self.source_conditioning = True
        self.source_preprocess = config.source_preprocess
        self.width = config.channels[0]
        self.stem = GroupConv(4, self.width, lifting=True, rngs=rngs)
        self.stem_modulation = GroupFiLM(self.width, rngs=rngs)
        self.blocks = nnx.List(
            [GroupBlock(self.width, d, rngs=rngs) for d in config.local_dilations]
        )
        self.head = GroupHead(self.width, rngs=rngs)

    def __call__(
        self,
        image,
        noise_level,
        insert,
        guide,
        progress,
        *,
        pyramid_level=0,
        return_network_output=False,
    ):
        assert return_network_output
        assert image.shape == noise_level.shape == guide.shape == progress.shape
        up = nearest_double(insert)
        features = jnp.concatenate((guide - 1, guide - 1, noise_level, up - 1), -1)
        basis = progress_basis(jnp.full_like(progress, pyramid_level / 5))
        x = nnx.silu(self.stem_modulation(self.stem(features), basis))
        for block in self.blocks:
            x = block(x, basis)
        return self.head(x)
