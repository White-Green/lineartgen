"""Exact square symmetries for inputs, masks, Gaussian draws and geometry choices."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


@partial(jax.jit, static_argnames=("inverse",))
def transform_batch(x, codes, *, inverse=False):
    assert x.ndim == 4

    def action(a, code):
        turns, mirror = code % 4, code // 4
        if inverse:
            a = jnp.rot90(a, -turns, axes=(0, 1))
            return jnp.flip(a, axis=1) if mirror else a
        a = jnp.flip(a, axis=1) if mirror else a
        return jnp.rot90(a, turns, axes=(0, 1))

    branches = tuple(lambda a, code=code: action(a, code) for code in range(8))
    return jax.vmap(lambda a, c: jax.lax.switch(c, branches, a))(x, codes)


def transform_numpy_batch(x, codes, *, inverse=False):
    assert x.ndim == 4
    result = []
    for a, code in zip(x, codes, strict=True):
        turns, mirror = int(code) % 4, int(code) // 4
        if inverse:
            a = np.rot90(a, -turns, axes=(0, 1))
            a = np.flip(a, axis=1) if mirror else a
        else:
            a = np.flip(a, axis=1) if mirror else a
            a = np.rot90(a, turns, axes=(0, 1))
        result.append(a)
    return np.stack(result)


def transform_directions(directions, codes, *, inverse=False):
    result = []
    for v, code in zip(directions, codes, strict=True):
        x, y = v
        turns, mirror = int(code) % 4, int(code) // 4
        if inverse:
            for _ in range(turns):
                x, y = -y, x
            x = -x if mirror else x
        else:
            x = -x if mirror else x
            for _ in range(turns):
                x, y = y, -x
        result.append((x, y))
    value = np.asarray(result, dtype=directions.dtype)
    return np.where(value == 0, np.zeros_like(value), value)


def transform_prepared(prepared, codes, *, inverse=False):
    return jax.tree.map(lambda x: transform_batch(x, codes, inverse=inverse), prepared)
