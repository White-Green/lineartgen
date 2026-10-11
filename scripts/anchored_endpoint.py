"""Bounded endpoint flow: isolate noisy versus original-source residual anchors."""

from functools import partial

import jax.numpy as jnp
from flax import nnx
from interval_velocity_fit import interval_error

MIN_REMAINING = 1 / 32


@partial(nnx.jit, static_argnames=("source_anchor",))
def endpoints(model, images, levels, guides, progress, source_anchor=False):
    smallest = images[-1]
    insert = jnp.zeros(
        (smallest.shape[0], smallest.shape[1] // 2, smallest.shape[2] // 2, 1), smallest.dtype
    )
    physical, bounded = [], []
    for x, t, s, u in zip(
        reversed(images), reversed(levels), reversed(guides), reversed(progress), strict=True
    ):
        head = model(x, t, insert, s, u, return_network_output=True)
        base = s if source_anchor else x
        raw = jnp.where(t > 0, base + jnp.sqrt(t) * head, s)
        insert = jnp.where(t > 0, jnp.clip(raw, -1, 1), s)
        physical.append(raw)
        bounded.append(insert)
    return tuple(reversed(physical)), tuple(reversed(bounded))


def velocity(model, images, levels, guides, progress, source_anchor=False):
    _, y = endpoints(model, images, levels, guides, progress, source_anchor)
    return tuple(
        jnp.where(t > 0, (a - x) / jnp.maximum(u, MIN_REMAINING), 0)
        for a, x, t, u in zip(y, images, levels, progress, strict=True)
    )


def objective(model, p, source_anchor=False):
    raw = endpoints(model, p["images"], p["levels"], p["guides"], p["progress"], source_anchor)[0][
        0
    ]
    error = interval_error(raw, p["target"], p["target"])
    active = p["levels"][0] > 0
    denom = jnp.maximum(active.sum((1, 2, 3)), 1)
    endpoint_mse = ((error**2 * active).sum((1, 2, 3)) / denom).mean()
    scaled = error / jnp.maximum(p["progress"][0], MIN_REMAINING)
    loss = ((scaled**2 * active).sum((1, 2, 3)) / denom).mean()
    return loss, dict(loss=loss, interval_endpoint_mse=endpoint_mse)


@nnx.jit
def noisy_gradient(model, p):
    (_, metrics), grads = nnx.value_and_grad(lambda m: objective(m, p, False), has_aux=True)(model)
    return metrics, grads


@nnx.jit
def source_gradient(model, p):
    (_, metrics), grads = nnx.value_and_grad(lambda m: objective(m, p, True), has_aux=True)(model)
    return metrics, grads
