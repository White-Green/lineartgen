"""Fixed-weight interventions on recursive subpixel geometry transfer."""

import jax.numpy as jnp
from flax import nnx
from subpixel_field_model import DISTANCE_RANGE, coverage, unpack

from lineartgen.diffusion import source_guides
from lineartgen.models.unet import progress_basis
from lineartgen.models.upsample import nearest_double

MODES = ("legacy", "scaled", "subpixel", "subpixel_scaled", "zero")


def pool_subpixels(field):
    """Average 2x2 subpixel cells onto the next stage's pixel centers."""
    b, h, w, c = field.shape
    return field.reshape(b, h // 2, 2, w // 2, 2, c).mean((2, 4))


def transfer(head, mode):
    assert mode in MODES
    if mode == "zero":
        return jnp.zeros((head.shape[0], head.shape[1] * 2, head.shape[2] * 2, 1), head.dtype)
    gain = 2 if mode in ("scaled", "subpixel_scaled") else 1
    if mode in ("legacy", "scaled"):
        return nearest_double(jnp.clip(head.mean(-1, keepdims=True) * gain, -1, 1))
    return jnp.clip(pool_subpixels(unpack(head)) * gain, -1, 1)


def field_forward(model, prepared, mode):
    guides = source_guides(model, prepared.guides)
    up = jnp.zeros_like(guides[-1])
    outputs = [None] * len(guides)
    for level in reversed(range(len(guides))):
        guide = guides[level]
        features = jnp.concatenate((guide - 1, guide - 1, prepared.noise_levels[level], up - 1), -1)
        basis = progress_basis(jnp.full_like(prepared.progress[level], level / 5))
        x = nnx.silu(model.stem_modulation(model.stem(features), basis))
        for block in model.blocks:
            x = block(x, basis)
        head = model.head(x)
        outputs[level] = unpack(head) * DISTANCE_RANGE
        if level:
            up = transfer(head, mode)
    return tuple(outputs)


@nnx.jit(static_argnames=("mode",))
def infer_transfer(model, prepared, *, mode):
    image = coverage(field_forward(model, prepared, mode)[0])
    return jnp.where(prepared.noise_levels[0] > 0, image, prepared.guides[0])
