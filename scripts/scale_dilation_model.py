"""Fixed-weight diagnostics of convolution reach at coarse recursion levels."""

import jax
import jax.numpy as jnp
from flax import nnx
from subpixel_field_model import DISTANCE_RANGE, coverage, unpack

from lineartgen.diffusion import source_guides
from lineartgen.models.dilated import dilated_same
from lineartgen.models.unet import progress_basis
from lineartgen.models.upsample import nearest_double

MODES = ("legacy", "physical", "bounded")


def dilation(base, level, shape, mode):
    assert mode in MODES
    if mode == "physical":
        return max(1, base // (2**level))
    if mode == "bounded":
        return min(base, max(1, (min(shape[1:3]) - 1) // 2))
    return base


def convolve(layer, x, level, mode):
    d = dilation(layer.dilation, level, x.shape, mode)
    kernel = layer.expanded_kernel()
    if d == 1:
        y = jax.lax.conv_general_dilated(
            x, kernel, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
        )
    else:
        y = dilated_same(x, kernel, d)
    return y + jnp.tile(layer.bias[...], 8)


def field_forward(model, prepared, mode):
    guides = source_guides(model, prepared.guides)
    g = guides[-1]
    insert = jnp.zeros((g.shape[0], g.shape[1] // 2, g.shape[2] // 2, 1), g.dtype)
    outputs = [None] * len(guides)
    for level in reversed(range(len(guides))):
        guide = guides[level]
        features = jnp.concatenate(
            (guide - 1, guide - 1, prepared.noise_levels[level], nearest_double(insert) - 1), -1
        )
        basis = progress_basis(jnp.full_like(prepared.progress[level], level / 5))
        x = nnx.silu(model.stem_modulation(convolve(model.stem, features, level, mode), basis))
        for block in model.blocks:
            y = nnx.silu(block.modulation(convolve(block.first, x, level, mode), basis))
            x = nnx.silu(x + convolve(block.second, y, level, mode))
        head = model.head(x)
        outputs[level] = unpack(head) * DISTANCE_RANGE
        insert = jnp.clip(head.mean(-1, keepdims=True), -1, 1)
    return tuple(outputs)


@nnx.jit(static_argnames=("mode",))
def infer_dilation(model, prepared, *, mode):
    image = coverage(field_forward(model, prepared, mode)[0])
    return jnp.where(prepared.noise_levels[0] > 0, image, prepared.guides[0])
