"""Resolution-first endpoint sampling and an explicitly teacher-assisted diagnostic."""

import jax.numpy as jnp
from flax import nnx
from velocity_bridge_toy import initialize

from lineartgen.diffusion import pyramid


@nnx.jit
def stage_velocity(model, x, t, s, u, insert):
    head = model(x, t, insert, s, u, return_network_output=True)
    raw = jnp.where(t > 0, x + jnp.sqrt(t) * head, s)
    bounded = jnp.where(t > 0, jnp.clip(raw, -1, 1), s)
    return jnp.where(t > 0, (bounded - x) / jnp.maximum(u, jnp.finfo(x.dtype).eps), 0)


def sample_cascade(model, source, strength, key, *, depth, steps=32, target=None, oracle=False):
    if steps != 32:
        raise ValueError("This diagnostic prescribes32 steps")
    if (target is not None) != oracle:
        raise ValueError("Only the explicitly teacher-assisted oracle accepts a target")
    images, levels, guides = initialize(source, strength, key, depth)
    small = images[-1]
    insert = jnp.zeros((small.shape[0], small.shape[1] // 2, small.shape[2] // 2, 1), small.dtype)
    indices = list(reversed(range(depth)))
    if oracle:
        clean = pyramid(jnp.where(strength > 0, target, source), depth)
        insert = clean[1] if depth > 1 else insert
        indices = [0]
    for i in indices:
        x, t, s = images[i], levels[i], guides[i]
        for n in range(steps):
            u = jnp.full_like(x, 1 - n / steps)
            v = stage_velocity(model, x, t, s, u, insert)
            x = jnp.where(t > 0, x + v / steps, s)
        insert = jnp.where(t > 0, jnp.clip(x, -1, 1), s)
    return x
