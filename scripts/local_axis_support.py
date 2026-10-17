"""Conservative permission for complete alternative strokes inside the active area."""

import jax.numpy as jnp
from consistent_axis_objective import candidate_mask, field_parts, preferred_original
from context_dropout_objective import dropout_forward
from flax import nnx
from subpixel_field_data import candidate_fields
from subpixel_field_model import SUBPIXELS


def support_safe(targets, strength):
    """Require nonempty ink in every candidate and its entire four-pixel support active.

    This repairs the constructed remote-mask contradiction. It does not establish
    local identifiability for arbitrary objects that intersect a mask boundary.
    """
    active = jnp.repeat(jnp.repeat(strength > 0, SUBPIXELS, 1), SUBPIXELS, 2)
    support = jnp.any(targets < 4, axis=1)
    nonempty = jnp.all(jnp.any(targets < -0.25, axis=(2, 3, 4)), axis=1)
    return nonempty & jnp.all(~support | active, axis=(1, 2, 3))


def support_mask(normal, targets, strength):
    safe = support_safe(targets, strength)
    original = preferred_original(normal) | ~safe
    return jnp.repeat(jnp.stack((original, ~original), 1), 9, 1)


def gradient_for(policy):
    assert policy in ("global", "local")

    @nnx.jit
    def gradient(model, prepared, payload, normal):
        base, axis, axis_normal, keep = payload
        depth = len(prepared.guides)
        aa = candidate_fields(base, normal, depth)
        bb = candidate_fields(axis, axis_normal, depth)
        # Identical costs and union ROI in both arms, even when an axis is forbidden.
        targets = tuple(jnp.concatenate((a, b), 1) for a, b in zip(aa, bb, strict=True))
        strength = prepared.noise_levels[0]
        safe = support_safe(targets[0], strength)
        mask = (
            candidate_mask(normal, strength, "consistent")
            if policy == "global"
            else support_mask(normal, targets[0], strength)
        )

        def objective(m):
            parts = field_parts(
                dropout_forward(m, prepared, keep),
                targets,
                prepared.noise_levels,
                candidate_mask=mask,
            )
            parts["support_safe_fraction"] = safe.astype(jnp.float32).mean()
            return parts["loss"], parts

        (_, parts), grads = nnx.value_and_grad(objective, has_aux=True)(model)
        return parts, grads

    return gradient
