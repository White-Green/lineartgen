"""Distance-dependent saturation bounds with exact antialias coverage targets."""

import jax.numpy as jnp

from .losses import tone_weights


def distance_margin_supervision(preclip, distance, clean, active, radius=8):
    error = preclip / (2 * radius) - distance / radius
    error = jnp.where(
        clean >= 1, jnp.minimum(error, 0), jnp.where(clean <= -1, jnp.maximum(error, 0), error)
    )
    error = jnp.where(error == 0, jnp.zeros_like(error), jnp.abs(error))
    weights = tone_weights(clean, 0.5) * active
    return ((error * weights).sum((1, 2, 3)) / jnp.maximum(weights.sum((1, 2, 3)), 1e-6)).mean()
