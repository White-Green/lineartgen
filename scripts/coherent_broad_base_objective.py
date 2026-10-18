"""HH: choose an observable, active-retaining axis from all valid isolated candidates."""

import jax
import jax.numpy as jnp
from coherent_mask_objective import axis_mask
from consistent_axis_objective import field_parts
from context_dropout_objective import dropout_forward
from flax import nnx
from forced_context_objective import family_parts
from opening_prior_objective import opening_consistency
from subpixel_field_data import candidate_fields


@jax.jit
def selected_original(base, normal, axis, axis_normal, strength, allowed):
    targets = jnp.concatenate(
        (candidate_fields(base, normal, 1)[0], candidate_fields(axis, axis_normal, 1)[0]), 1
    )
    return axis_mask(normal, targets, strength, "retained")[:, 0] | ~allowed


def allowed_mask(normal, allowed, strengths, candidates, policy):
    assert policy == "retained"
    chosen = axis_mask(normal, candidates, strengths[0], policy)
    original = chosen[:, 0] | ~allowed
    return jnp.repeat(jnp.stack((original, ~original), 1), 9, 1)


def axis_parts(fields, original, merged, axis, strengths, eligible, allowed, normal, *, policy):
    teacher = tuple(
        jnp.where(eligible[:, None, None, None, None], b, a)
        for a, b in zip(original, merged, strict=True)
    )
    candidates = tuple(
        jnp.concatenate((a, jnp.where(allowed[:, None, None, None, None], b, a)), 1)
        for a, b in zip(teacher, axis, strict=True)
    )

    def extended(_):
        parts = field_parts(
            fields,
            candidates,
            strengths,
            sign_power=4,
            candidate_mask=allowed_mask(normal, allowed, strengths, candidates[0], policy),
        )
        return parts | dict(
            merged_fraction=eligible.astype(jnp.float32).mean()
        ), opening_consistency(fields[0], candidates[0], strengths[0])

    def unchanged(_):
        parts = family_parts(fields, original, merged, strengths, eligible)
        parts |= dict(
            selected_alternate_fraction=jnp.float32(0), selected_alternate_bits=jnp.float32(0)
        )
        return parts, opening_consistency(fields[0], teacher[0], strengths[0])

    parts, prior = jax.lax.cond(jnp.any(allowed), extended, unchanged, operand=None)
    return parts | dict(
        opening_consistency=prior,
        loss=parts["loss"] + 0.0 * prior,
        axis_eligible_fraction=allowed.astype(jnp.float32).mean(),
    )


def axis_gradient(policy="retained"):
    @nnx.jit
    def gradient(model, prepared, bases, directions):
        base, merged, eligible, keep, axis, axis_directions, allowed = bases
        depth = len(prepared.guides)
        original = candidate_fields(base, directions, depth)
        alternative = candidate_fields(merged, directions, depth)
        perpendicular = candidate_fields(axis, axis_directions, depth)

        def objective(m):
            parts = axis_parts(
                dropout_forward(m, prepared, keep),
                original,
                alternative,
                perpendicular,
                prepared.noise_levels,
                eligible,
                allowed,
                directions,
                policy=policy,
            )
            return parts["loss"], parts

        (_, parts), grads = nnx.value_and_grad(objective, has_aux=True)(model)
        return parts, grads

    return gradient
