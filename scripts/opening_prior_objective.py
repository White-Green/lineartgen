"""Penalize narrow ink unsupported by any permitted teacher geometry."""

import jax
import jax.numpy as jnp
from context_dropout_objective import dropout_forward
from flax import nnx
from forced_context_objective import family_parts, forced_gradient
from subpixel_field_data import candidate_fields
from subpixel_field_model import SUBPIXELS

OFFSETS = tuple((y, x) for y in range(-2, 3) for x in range(-2, 3) if x * x + y * y <= 4)


def disk_reduce(values, *, maximum):
    """Radius0.5 native pixel on the4x subpixel lattice; edge extension."""
    assert values.ndim == 4
    h, w = values.shape[1:3]
    padded = jnp.pad(values, ((0, 0), (2, 2), (2, 2), (0, 0)), mode="edge")
    result = values
    op = jnp.maximum if maximum else jnp.minimum
    for y, x in OFFSETS:
        result = op(result, padded[:, 2 + y : 2 + y + h, 2 + x : 2 + x + w])
    return result


def soft_opening(alpha):
    return disk_reduce(disk_reduce(alpha, maximum=False), maximum=True)


def opening_consistency(pred, target, strengths):
    """Fine-stage only. Teacher residual maximum exempts all nine valid shifts."""
    b, k, h, w, c = target.shape
    assert pred.shape == (b, h, w, c)
    pa = jnp.clip(0.5 - SUBPIXELS * pred, 0, 1)
    ta = jnp.clip(0.5 - SUBPIXELS * target, 0, 1)
    opened_target = soft_opening(ta.reshape(b * k, h, w, c)).reshape(ta.shape)
    allowance = jnp.maximum(ta - opened_target, 0).max(1)
    residual = jax.nn.relu(pa - soft_opening(pa) - allowance)
    active = jnp.repeat(jnp.repeat(strengths > 0, SUBPIXELS, 1), SUBPIXELS, 2).astype(pred.dtype)
    # Exclude any center whose two successive disk neighborhoods cross the inactive region.
    valid = disk_reduce(disk_reduce(active, maximum=False), maximum=False)
    denominator = jnp.maximum((ta * valid[:, None]).sum((2, 3, 4)).max(1), 1)
    mean_square = (residual * residual * valid).sum((1, 2, 3)) / denominator
    return (jnp.sqrt(mean_square + 0.01**2) - 0.01).mean()


def opening_gradient(*, weight=0.1):
    assert weight >= 0
    if weight == 0:
        return forced_gradient()

    @nnx.jit
    def gradient(model, prepared, bases, directions):
        base, merged, eligible, keep = bases
        original = candidate_fields(base, directions, len(prepared.guides))
        alternative = candidate_fields(merged, directions, len(prepared.guides))
        teacher = jnp.where(eligible[:, None, None, None, None], alternative[0], original[0])

        def objective(m):
            fields = dropout_forward(m, prepared, keep)
            parts = family_parts(fields, original, alternative, prepared.noise_levels, eligible)
            prior = opening_consistency(fields[0], teacher, prepared.noise_levels[0])
            parts = parts | dict(opening_consistency=prior, loss=parts["loss"] + weight * prior)
            return parts["loss"], parts

        (_, parts), grads = nnx.value_and_grad(objective, has_aux=True)(model)
        return parts, grads

    return gradient
