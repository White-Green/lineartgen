"""HP: linear white-margin repair beyond every permissible teacher support."""

import jax
import jax.numpy as jnp
from coherent_broad_objective import axis_parts
from context_dropout_objective import dropout_forward
from flax import nnx
from subpixel_field_data import candidate_fields
from subpixel_field_model import SUBPIXELS


def support_margin(pred, targets, strength):
    """Fine stage only; all candidate shifts/axes are exempt within one pixel.

    The normalization is teacher-only. Saturated black and faint false ink get
    the same inward field derivative; correct white and protected samples get0.
    This is not a topology guarantee or a penalty on legitimate antialiasing.
    """
    active = jnp.repeat(jnp.repeat(strength > 0, SUBPIXELS, 1), SUBPIXELS, 2)
    exterior = (targets.min(1) > 1.0) & active
    alpha = jnp.clip(0.5 - SUBPIXELS * targets, 0, 1)
    ink = (alpha * active[:, None]).sum((2, 3, 4)).max(1)
    denominator = ink + 0.001 * active.sum((1, 2, 3)) + 1e-6
    value = (jax.nn.relu(0.25 - pred) * exterior).sum((1, 2, 3)) / denominator
    return value.mean()


def support_parts(
    fields,
    original,
    alternative,
    perpendicular,
    strengths,
    eligible,
    allowed,
    directions,
    *,
    weight,
):
    parts = axis_parts(
        fields,
        original,
        alternative,
        perpendicular,
        strengths,
        eligible,
        allowed,
        directions,
        policy="retained",
        opening_weight=0.1,
    )
    teacher = jnp.where(eligible[:, None, None, None, None], alternative[0], original[0])
    candidates = jnp.concatenate(
        (teacher, jnp.where(allowed[:, None, None, None, None], perpendicular[0], teacher)), axis=1
    )
    prior = support_margin(fields[0], candidates, strengths[0])
    return parts | dict(support_margin=prior, loss=parts["loss"] + weight * prior)


def support_gradient(*, weight):
    assert weight >= 0

    @nnx.jit
    def gradient(model, prepared, bases, directions):
        base, merged, eligible, keep, axis, axis_directions, allowed = bases
        depth = len(prepared.guides)
        original = candidate_fields(base, directions, depth)
        alternative = candidate_fields(merged, directions, depth)
        perpendicular = candidate_fields(axis, axis_directions, depth)

        def objective(m):
            parts = support_parts(
                dropout_forward(m, prepared, keep),
                original,
                alternative,
                perpendicular,
                prepared.noise_levels,
                eligible,
                allowed,
                directions,
                weight=weight,
            )
            return parts["loss"], parts

        (_, parts), grads = nnx.value_and_grad(objective, has_aux=True)(model)
        return parts, grads

    return gradient
