"""Source-centered, variance-preconditioned velocity for the same linear bridge."""

import jax.numpy as jnp
from flax import nnx

SIGNAL_STD = 0.5


def coefficients(level, remaining):
    # Protected pixels use a finite dummy variance; their state and velocity are
    # explicitly projected below. No division by remaining time is required.
    sigma = jnp.where(level > 0, 0.25 * level, 1.0)
    tau = 1 - remaining
    variance = tau**2 * SIGNAL_STD**2 + remaining**2 * sigma**2
    scale = jnp.sqrt(variance)
    skip = (tau * SIGNAL_STD**2 - remaining * sigma**2) / variance
    output = SIGNAL_STD * sigma / scale
    return 1 / scale, skip, output


@nnx.jit
def recursive_velocity(model, images, levels, guides, progress):
    smallest = images[-1]
    insert = jnp.zeros(
        (smallest.shape[0], smallest.shape[1] // 2, smallest.shape[2] // 2, 1), smallest.dtype
    )
    velocities, endpoints = [], []
    for x, t, s, u in zip(
        reversed(images), reversed(levels), reversed(guides), reversed(progress), strict=True
    ):
        cin, skip, cout = coefficients(t, u)
        network_input = jnp.where(t > 0, (x - s) * cin, 0)
        # Read F before the usual residual anchor/scale, without dividing them out.
        raw = model(network_input, t, insert, s, u, return_network_output=True)
        v = jnp.where(t > 0, skip * (x - s) + cout * raw, 0)
        endpoint = jnp.where(t > 0, x + u * v, s)
        insert = jnp.clip(endpoint, -1, 1)
        velocities.append(v)
        endpoints.append(endpoint)
    return tuple(reversed(velocities)), tuple(reversed(endpoints))


def velocity(model, images, levels, guides, progress):
    return recursive_velocity(model, images, levels, guides, progress)[0]


def objective(model, prepared, *, normalized=False):
    v = velocity(
        model, prepared["images"], prepared["levels"], prepared["guides"], prepared["progress"]
    )[0]
    error = v - prepared["velocity_target"]
    active = prepared["levels"][0] > 0
    denom = jnp.maximum(active.sum((1, 2, 3)), 1)
    physical = ((error**2 * active).sum((1, 2, 3)) / denom).mean()
    if normalized:
        cout = coefficients(prepared["levels"][0], prepared["progress"][0])[2]
        error = error / cout
    loss = ((error**2 * active).sum((1, 2, 3)) / denom).mean()
    return loss, dict(loss=loss, physical_velocity_mse=physical)


@nnx.jit
def physical_gradient(model, prepared):
    (_, metrics), grads = nnx.value_and_grad(lambda m: objective(m, prepared), has_aux=True)(model)
    return metrics, grads


@nnx.jit
def normalized_gradient(model, prepared):
    (_, metrics), grads = nnx.value_and_grad(
        lambda m: objective(m, prepared, normalized=True), has_aux=True
    )(model)
    return metrics, grads
