"""Local soft distance fields for grayscale line boundaries.

Repeated square dilation gives a truncated Chebyshev distance for binary masks.
The same max/mean operations extend it continuously to antialiased ink. This is
an experimental distance-field matching loss, not the linear signed-distance
boundary loss or an exact Hausdorff distance.
"""

import jax
import jax.numpy as jnp


def soft_distance(occupancy, radius):
    """Average absence over dilation radii 0..radius-1; normalized to [0, 1]."""
    expanded = occupancy
    distance = 1 - expanded
    for _ in range(1, radius):
        # The max identity selects JAX's differentiable max-pool primitive.
        expanded = jax.lax.reduce_window(
            expanded, -jnp.inf, jax.lax.max, (1, 3, 3, 1), (1, 1, 1, 1), "SAME"
        )
        distance += 1 - expanded
    return distance / radius


def boundary_distance_loss(prediction, target, active, radius):
    """Match ink and paper distances in a target boundary band, preserving soft targets.

    Protected input values enter both fields identically, without prediction gradients.
    Blank/full targets have no boundary band and leave supervision to the base loss.
    The radius is in pixels of the current scale; recursion loss weights are external.
    """
    truth = jnp.clip((1 - target) / 2, 0, 1)
    predicted = jnp.clip((1 - prediction) / 2, 0, 1)
    predicted = jnp.where(active, predicted, jax.lax.stop_gradient(truth))
    reference = jax.lax.stop_gradient(
        jnp.concatenate((soft_distance(truth, radius), soft_distance(1 - truth, radius)), axis=-1)
    )
    estimated = jnp.concatenate(
        (soft_distance(predicted, radius), soft_distance(1 - predicted, radius)), axis=-1
    )
    band = jnp.all(reference < 1 - 1e-6, axis=-1, keepdims=True) & active
    error = jnp.mean(jnp.square(estimated - reference), axis=-1, keepdims=True)
    axes = (1, 2, 3)
    count = band.sum(axes)
    per_image = (error * band).sum(axes) / jnp.maximum(count, 1)
    return per_image.mean()
