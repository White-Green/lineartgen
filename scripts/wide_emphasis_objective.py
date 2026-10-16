"""Importance weighting of broad examples; keep within-example geometry unchanged."""

import jax
import jax.numpy as jnp
from context_dropout_objective import dropout_forward
from emphasized_field_objective import field_parts
from flax import nnx
from opening_prior_objective import opening_consistency, opening_gradient
from subpixel_field_data import candidate_fields

FACTOR = 5.0
NORMALIZATION = 1.204  # 1 + (5-1)*1224/24000, fixed from the predeclared source plan.


def emphasized_parts(fields, original, merged, strengths, eligible, weights):
    selected = tuple(
        jnp.where(eligible[:, None, None, None, None], b, a)
        for a, b in zip(original, merged, strict=True)
    )
    parts = field_parts(fields, selected, strengths, example_weights=weights, sign_power=4)
    prior = jax.vmap(lambda p, t, s: opening_consistency(p[None], t[None], s[None]))(
        fields[0], selected[0], strengths[0]
    )
    prior = (prior * weights).mean()
    return parts | dict(
        opening_consistency=prior,
        merged_fraction=eligible.astype(jnp.float32).mean(),
        loss=parts["loss"] + 0.1 * prior,
    )


def emphasis_gradient(*, enabled):
    if not enabled:
        gradient = opening_gradient()
        return lambda model, prepared, bases, directions: gradient(
            model, prepared, bases[:4], directions
        )

    @nnx.jit
    def gradient(model, prepared, bases, directions):
        base, merged, eligible, keep, weights = bases
        original = candidate_fields(base, directions, len(prepared.guides))
        alternative = candidate_fields(merged, directions, len(prepared.guides))

        def objective(m):
            parts = emphasized_parts(
                dropout_forward(m, prepared, keep),
                original,
                alternative,
                prepared.noise_levels,
                eligible,
                weights,
            )
            return parts["loss"], parts

        (_, parts), grads = nnx.value_and_grad(objective, has_aux=True)(model)
        return parts, grads

    return gradient
