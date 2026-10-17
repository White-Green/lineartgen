"""Recipe-independent choices for two observationally equivalent stroke axes."""

import jax.numpy as jnp
from consistent_axis_objective import field_parts, preferred_original
from context_dropout_objective import dropout_forward
from flax import nnx
from local_axis_support import support_mask, support_safe
from subpixel_field_data import candidate_fields
from subpixel_field_model import SUBPIXELS


def axis_mask(normal, targets, strength, policy):
    assert policy in ("guard", "canonical", "retained")
    if policy == "guard":
        return support_mask(normal, targets, strength)
    original = preferred_original(normal)
    if policy == "retained":
        active = jnp.repeat(jnp.repeat(strength > 0, SUBPIXELS, 1), SUBPIXELS, 2)
        ink = targets < -0.25
        fraction = (ink * active[:, None]).sum((2, 3, 4)) / jnp.maximum(ink.sum((2, 3, 4)), 1)
        scores = fraction.reshape(len(normal), 2, 9).mean(2)
        difference = scores[:, 0] - scores[:, 1]
        original = jnp.where(jnp.abs(difference) <= 1e-6, original, difference > 0)
    return jnp.repeat(jnp.stack((original, ~original), 1), 9, 1)


def gradient_for(policy):
    @nnx.jit
    def gradient(model, prepared, payload, normal):
        base, axis, axis_normal, keep = payload
        depth = len(prepared.guides)
        aa = candidate_fields(base, normal, depth)
        bb = candidate_fields(axis, axis_normal, depth)
        targets = tuple(jnp.concatenate((a, b), 1) for a, b in zip(aa, bb, strict=True))
        strength = prepared.noise_levels[0]
        mask = axis_mask(normal, targets[0], strength, policy)
        safe = support_safe(targets[0], strength)

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
