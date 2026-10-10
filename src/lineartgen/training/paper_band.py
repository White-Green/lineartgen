"""Penalize excess ink in teacher paper close to a line, including open contours."""

import jax
import jax.numpy as jnp


def paper_band_importance(target, active, radius=3):
    alpha = jax.lax.stop_gradient(jnp.clip((1 - target) / 2, 0, 1))
    window = (1, 2 * radius + 1, 2 * radius + 1, 1)
    dilated = jax.lax.reduce_window(alpha, -jnp.inf, jax.lax.max, window, (1, 1, 1, 1), "SAME")
    weight = jnp.where(
        active & (alpha <= 0.02) & (dilated > 0.25), jax.nn.relu(dilated - alpha), 0.0
    )
    return jax.lax.stop_gradient(weight)


def paper_band_loss(preclip, target, active, radius=3):
    """Use physical preclip values so falsely saturated ink still receives a gradient.

    Teacher ink above .02 and protected pixels receive no direct gradient. The
    mask includes narrow gaps; an enabled gap loss adds deliberate extra emphasis
    there. Shared-parameter changes can still affect real ink elsewhere.
    """
    weights = paper_band_importance(target, active, radius)
    truth = jax.lax.stop_gradient(jnp.clip((1 - target) / 2, 0, 1))
    excess = jax.nn.relu((1 - preclip) / 2 - truth)
    axes = (1, 2, 3)
    return ((weights * excess).sum(axes) / jnp.maximum(weights.sum(axes), 1e-6)).mean()
