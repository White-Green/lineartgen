"""Strict named-array transfer of the training-only critic, without optimizer state."""

import json

import jax.numpy as jnp
import numpy as np
from flax import nnx

from lineartgen.models.serialization import _arrays


def save_prior(critic, path):
    state = nnx.to_pure_dict(nnx.state(critic, nnx.Param))
    np.savez(path, **{key: np.asarray(value) for key, value in _arrays(state)})


def load_prior(critic, path):
    state = nnx.state(critic, nnx.Param)
    template = nnx.to_pure_dict(state)
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != {key for key, _ in _arrays(template)}:
            raise ValueError("Critic weight keys do not match")

        def restore(tree, route=()):
            if isinstance(tree, dict):
                return {k: restore(v, (*route, k)) for k, v in tree.items()}
            value = arrays[json.dumps(route)]
            if value.shape != tree.shape or value.dtype != tree.dtype:
                raise ValueError("Critic weight shape/dtype does not match")
            return jnp.asarray(value)

        nnx.replace_by_pure_dict(state, restore(template))
    nnx.update(critic, state)
