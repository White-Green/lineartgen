"""Train every scale while independently dropping per-example recursive edges."""

import jax.numpy as jnp
import numpy as np
from flax import nnx
from merge_field_objective import family_parts
from subpixel_field_data import candidate_fields
from subpixel_field_model import DISTANCE_RANGE, unpack

from lineartgen.diffusion import source_guides


def context_mask(step, depth, batch, *, enabled):
    keep = np.ones((depth, batch, 1, 1, 1), np.float32)
    if enabled and depth > 1:
        rng = np.random.default_rng([2026101505, step, 0x435458])
        keep[1:] = (rng.random((depth - 1, batch, 1, 1, 1)) >= 0.5).astype(np.float32)
    return keep


def dropout_forward(model, prepared, keep):
    guides = source_guides(model, prepared.guides)
    g = guides[-1]
    insert = jnp.zeros((g.shape[0], g.shape[1] // 2, g.shape[2] // 2, 1), g.dtype)
    outputs = [None] * len(guides)
    for level in reversed(range(len(guides))):
        head = model(
            prepared.guides[level]
            if getattr(model, "uses_raw_guide", False)
            else prepared.noisy[level],
            prepared.noise_levels[level],
            insert,
            guides[level],
            prepared.progress[level],
            pyramid_level=level,
            return_network_output=True,
        )
        outputs[level] = unpack(head) * DISTANCE_RANGE
        insert = jnp.clip(head.mean(-1, keepdims=True), -1, 1) * keep[level]
    return tuple(outputs)


def dropout_gradient():
    @nnx.jit
    def gradient(model, prepared, bases, directions):
        base, merged, eligible, keep = bases
        original = candidate_fields(base, directions, len(prepared.guides))
        alternative = candidate_fields(merged, directions, len(prepared.guides))

        def objective(m):
            parts = family_parts(
                dropout_forward(m, prepared, keep),
                original,
                alternative,
                prepared.noise_levels,
                eligible,
            )
            return parts["loss"], parts

        (_, parts), grads = nnx.value_and_grad(objective, has_aux=True)(model)
        return parts, grads

    return gradient
