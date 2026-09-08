"""Exact one-to-one noise assignment with Burn's masked corruption cost.

For image x_i, noise level t_i and candidate z_j the old cost is
sum(adaptive_avg_pool((x_i - sqrt(1-t_i)*x_i - sqrt(t_i)*z_j)**2)).
Expanding that quadratic replaces a [batch, pool, H, W, C] temporary with two
matrix products. Pooling after squaring is a weighted sum, not a low-pass
comparison of the images. The objective is unchanged, up to floating point.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.optimize import linear_sum_assignment


def _pool_sum_weights(size, resolution):
    """Weights equivalent to sum(adaptive_avg_pool), including overlapping bins."""
    count = min(size, resolution)
    weights = np.zeros(size, np.float32)
    for index in range(count):
        start = index * size // count
        end = ((index + 1) * size + count - 1) // count
        weights[start:end] += 1 / (end - start)
    return weights


def pairwise_noise_costs(clean, noise_level, candidates, *, resolution=32):
    if (
        clean.ndim != 4
        or noise_level.shape != clean.shape
        or candidates.shape[1:] != clean.shape[1:]
    ):
        raise ValueError("Matching expects NHWC clean/noise maps and same-size candidates")
    _, height, width, channels = clean.shape
    weights = (
        _pool_sum_weights(height, resolution)[:, None]
        * _pool_sum_weights(width, resolution)[None, :]
    )
    weights = jnp.broadcast_to(jnp.asarray(weights)[..., None], (height, width, channels)).reshape(
        -1
    )
    clean, noise_level, candidates = jax.tree.map(
        jax.lax.stop_gradient, (clean, noise_level, candidates)
    )
    residual = (clean * (1 - jnp.sqrt(1 - noise_level))).reshape(clean.shape[0], -1)
    level = noise_level.reshape(clean.shape[0], -1)
    noise = candidates.reshape(candidates.shape[0], -1)
    constant = (residual**2 * weights).sum(axis=1, keepdims=True)
    norm = jnp.matmul(level * weights, (noise**2).T, precision=jax.lax.Precision.HIGHEST)
    cross = jnp.matmul(
        residual * jnp.sqrt(level) * weights, noise.T, precision=jax.lax.Precision.HIGHEST
    )
    return constant + norm - 2 * cross


def _scipy_assignment(costs):
    return linear_sum_assignment(np.asarray(costs))[1].astype(np.int32)


def assignment_indices(costs, *, solver="scipy"):
    if costs.ndim != 2 or costs.shape[1] < costs.shape[0] or min(costs.shape) < 1:
        raise ValueError("Assignment requires a nonempty pool at least as large as the batch")
    costs = jax.lax.stop_gradient(costs)
    if solver == "scipy":
        # Only the small [batch, pool] matrix crosses the device boundary.
        # This callback is part of input preparation, outside model gradients.
        return jax.pure_callback(
            _scipy_assignment,
            jax.ShapeDtypeStruct((costs.shape[0],), jnp.int32),
            costs,
            vmap_method="sequential",
        )
    if solver != "optax":
        raise ValueError(f"Unknown exact assignment solver: {solver}")
    rows, columns = optax.assignment.hungarian_algorithm(costs)
    # Do not rely on the solver's output order.
    return jnp.zeros(costs.shape[0], dtype=jnp.int32).at[rows].set(columns)


def match_noise(clean, noise_level, candidates, *, resolution=32, solver="scipy"):
    costs = pairwise_noise_costs(clean, noise_level, candidates, resolution=resolution)
    return candidates[assignment_indices(costs, solver=solver)]
