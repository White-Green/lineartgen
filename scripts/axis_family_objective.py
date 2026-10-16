"""Keep a whole-field choice between supported orientations instead of averaging them."""

import jax
import jax.numpy as jnp
from context_dropout_objective import dropout_forward
from flax import nnx
from forced_context_objective import family_parts
from opening_prior_objective import opening_consistency
from quartic_field_objective import field_parts
from subpixel_field_data import candidate_fields


def axis_parts(fields, original, merged, axis, strengths, eligible, allowed):
    allowed = allowed & jnp.all(strengths[0] > 0, axis=(1, 2, 3))
    teacher = tuple(
        jnp.where(eligible[:, None, None, None, None], b, a)
        for a, b in zip(original, merged, strict=True)
    )
    candidates = tuple(
        jnp.concatenate((a, jnp.where(allowed[:, None, None, None, None], b, a)), 1)
        for a, b in zip(teacher, axis, strict=True)
    )

    def extended(_):
        parts = field_parts(fields, candidates, strengths, sign_power=4)
        return parts | dict(
            merged_fraction=eligible.astype(jnp.float32).mean()
        ), opening_consistency(fields[0], candidates[0], strengths[0])

    def unchanged(_):
        return family_parts(fields, original, merged, strengths, eligible), opening_consistency(
            fields[0], teacher[0], strengths[0]
        )

    parts, prior = jax.lax.cond(jnp.any(allowed), extended, unchanged, operand=None)
    return parts | dict(
        opening_consistency=prior,
        loss=parts["loss"] + 0.1 * prior,
        axis_eligible_fraction=allowed.astype(jnp.float32).mean(),
    )


def axis_gradient():
    @nnx.jit
    def gradient(model, prepared, bases, directions):
        base, merged, eligible, keep, axis, axis_directions, allowed = bases
        depth = len(prepared.guides)
        original = candidate_fields(base, directions, depth)
        alternative = candidate_fields(merged, directions, depth)
        perpendicular = candidate_fields(axis, axis_directions, depth)

        def objective(m):
            fields = dropout_forward(m, prepared, keep)
            parts = axis_parts(
                fields,
                original,
                alternative,
                perpendicular,
                prepared.noise_levels,
                eligible,
                allowed,
            )
            return parts["loss"], parts

        (_, parts), grads = nnx.value_and_grad(objective, has_aux=True)(model)
        return parts, grads

    return gradient
