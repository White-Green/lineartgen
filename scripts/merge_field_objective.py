"""Select one complete opaque geometry family per image across all scales."""

import jax
import jax.numpy as jnp
from flax import nnx
from regional_field_objective import field_parts
from subpixel_field_data import candidate_fields
from subpixel_field_model import field_forward


def family_parts(fields, original, merged, strengths, eligible):
    def ordinary(_):
        return field_parts(fields, original, strengths) | dict(merged_fraction=jnp.float32(0))

    def alternatives(_):
        # Each family retains the inherited normalization over its nine shifts.
        # A merged path cannot reduce its loss by diluting errors across a larger ROI.
        def per_image(ff, tt, ss):
            return field_parts(
                tuple(x[None] for x in ff),
                tuple(x[None] for x in tt),
                tuple(x[None] for x in ss),
            )

        a = jax.vmap(per_image)(fields, original, strengths)
        b = jax.vmap(per_image)(fields, merged, strengths)
        choose = eligible & (b["loss"] < a["loss"])
        return {k: jnp.where(choose, b[k], a[k]).mean() for k in a} | dict(
            merged_fraction=choose.astype(jnp.float32).mean()
        )

    return jax.lax.cond(jnp.any(eligible), alternatives, ordinary, operand=None)


def merged_gradient():
    @nnx.jit
    def gradient(model, prepared, bases, directions):
        base, merged, eligible = bases
        original_targets = candidate_fields(base, directions, len(prepared.guides))
        merged_targets = candidate_fields(merged, directions, len(prepared.guides))

        def objective(m):
            parts = family_parts(
                field_forward(m, prepared),
                original_targets,
                merged_targets,
                prepared.noise_levels,
                eligible,
            )
            return parts["loss"], parts

        (_, parts), grads = nnx.value_and_grad(objective, has_aux=True)(model)
        return parts, grads

    return gradient
