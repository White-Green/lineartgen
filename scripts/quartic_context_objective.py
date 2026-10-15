"""Keep geometry families and recursive dropout; change only signed-error aggregation."""

import jax
import jax.numpy as jnp
from context_dropout_objective import dropout_forward
from flax import nnx
from quartic_field_objective import field_parts
from subpixel_field_data import candidate_fields


def family_parts(fields, original, merged, strengths, eligible, *, sign_power=4):
    def ordinary(_):
        return field_parts(fields, original, strengths, sign_power=sign_power) | dict(
            merged_fraction=jnp.float32(0)
        )

    def alternatives(_):
        # Each family retains the inherited normalization over its nine shifts.
        # A merged path cannot reduce its loss by diluting errors across a larger ROI.
        def per_image(ff, tt, ss):
            return field_parts(
                tuple(x[None] for x in ff),
                tuple(x[None] for x in tt),
                tuple(x[None] for x in ss),
                sign_power=sign_power,
            )

        a = jax.vmap(per_image)(fields, original, strengths)
        b = jax.vmap(per_image)(fields, merged, strengths)
        choose = eligible & (b["loss"] < a["loss"])
        return {k: jnp.where(choose, b[k], a[k]).mean() for k in a} | dict(
            merged_fraction=choose.astype(jnp.float32).mean()
        )

    return jax.lax.cond(jnp.any(eligible), alternatives, ordinary, operand=None)


def quartic_gradient(*, sign_power=4):
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
                sign_power=sign_power,
            )
            return parts["loss"], parts

        (_, parts), grads = nnx.value_and_grad(objective, has_aux=True)(model)
        return parts, grads

    return gradient
