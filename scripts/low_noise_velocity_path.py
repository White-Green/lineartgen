"""Same BC bridge and Euler nodes, with a configurable initial noise amplitude."""

import functools

import jax
import jax.numpy as jnp
from bridge_time_grid import time_nodes
from geometry_velocity_fit import SEED

from lineartgen.diffusion import pyramid

NOISE_SCALE = 0.05


def initialize(source, strength, key, depth, noise_scale):
    guides, levels = pyramid(source, depth), pyramid(strength, depth)
    keys = jax.random.split(key, depth)
    noisy = tuple(
        s + noise_scale * t * jax.random.normal(k, s.shape)
        for s, t, k in zip(guides, levels, keys, strict=True)
    )
    return noisy, levels, guides


def path_inputs(source, target, strength, key, tau, depth, noise_scale):
    z, levels, guides = initialize(source, strength, key, depth, noise_scale)
    clean = pyramid(jnp.where(strength > 0, target, source), depth)
    images = tuple(
        jnp.where(t > 0, (1 - tau) * x + tau * y, s)
        for x, y, t, s in zip(z, clean, levels, guides, strict=True)
    )
    progress = tuple(jnp.broadcast_to(1 - tau, x.shape) for x in images)
    return dict(images=images, levels=levels, guides=guides, progress=progress, target=clean[0])


@functools.partial(jax.jit, static_argnames=("depth", "noise_scale"))
def prepared_batch(clean, source, strength, step, depth, noise_scale=NOISE_SCALE):
    key = jax.random.fold_in(jax.random.PRNGKey(SEED), step)
    draws = jax.random.uniform(jax.random.fold_in(key, 1), (len(clean), 2, 1, 1, 1))
    raw = jnp.where(draws[:, 0] < 0.25, 0, draws[:, 1])
    tau = jnp.square(jnp.sin(jnp.pi * raw / 2))
    noise_key = jax.random.fold_in(key, 2)
    direct = path_inputs(
        source, clean, strength, noise_key, jnp.zeros_like(tau), depth, noise_scale
    )
    bridge = path_inputs(source, clean, strength, noise_key, tau, depth, noise_scale)
    return dict(direct=direct, bridge=bridge), tau


def sample_grid(predict, source, strength, key, *, depth, steps, noise_scale):
    images, levels, guides = initialize(source, strength, key, depth, noise_scale)
    nodes = time_nodes(steps)
    for start, end in zip(nodes[:-1], nodes[1:], strict=True):
        progress = tuple(jnp.full_like(x, 1 - start) for x in images)
        estimates = predict(images, levels, guides, progress)
        images = tuple(
            jnp.where(t > 0, x + (end - start) * v, s)
            for x, v, t, s in zip(images, estimates, levels, guides, strict=True)
        )
    return images[0]
