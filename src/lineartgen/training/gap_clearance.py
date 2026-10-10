"""Retain narrow teacher-paper regions without directly penalizing teacher ink."""

import jax
import jax.numpy as jnp


def paper_importance(target, active, radius=3):
    alpha = jax.lax.stop_gradient(jnp.clip((1 - target) / 2, 0, 1))
    window = (1, 2 * radius + 1, 2 * radius + 1, 1)
    strides = (1, 1, 1, 1)
    dilated = jax.lax.reduce_window(alpha, -jnp.inf, jax.lax.max, window, strides, "SAME")
    closed = -jax.lax.reduce_window(-dilated, -jnp.inf, jax.lax.max, window, strides, "SAME")
    weight = jnp.where(active & (alpha <= 0.02) & (closed > 0.25), jax.nn.relu(closed - alpha), 0.0)
    return jax.lax.stop_gradient(weight)


def gap_clearance_loss(preclip, target, active, radius=3):
    """One-sided excess ink in teacher-paper gaps; physical values retain saturation gradients.

    Square closing can also identify small concave paper regions; this is not a
    topological guarantee or a detector based on the model's own prediction.
    """
    weights = paper_importance(target, active, radius)
    truth = jax.lax.stop_gradient(jnp.clip((1 - target) / 2, 0, 1))
    excess = jax.nn.relu((1 - preclip) / 2 - truth)
    axes = (1, 2, 3)
    return ((weights * excess).sum(axes) / jnp.maximum(weights.sum(axes), 1e-6)).mean()
