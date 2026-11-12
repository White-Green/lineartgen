"""Learned whole-component decisions from the final CNN state, without JAX."""

import numpy as np

from .selection_feedback import components

SHAPES = {
    "hidden_kernel": (32, 32),
    "hidden_bias": (32,),
    "kernel": (32, 1),
    "bias": (1,),
}


def load_parameters(path):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != SHAPES.keys() or len(archive.files) != len(SHAPES):
            raise ValueError("Invalid component readout parameter names")
        params = {name: archive[name] for name in SHAPES}
    for name, shape in SHAPES.items():
        value = params[name]
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"Invalid component readout parameter: {name}")
        value.flags.writeable = False
    return params


def descriptors(state, ids):
    if state.shape != (*ids.shape, 16) or not np.isfinite(state).all():
        raise ValueError("Expected finite HWC state and matching component labels")
    valid = ids >= 0
    labels, values = ids[valid], state[valid]
    count = np.bincount(labels)
    total = np.zeros((len(count), 16), np.float64)
    maximum = np.full((len(count), 16), -np.inf, np.float32)
    np.add.at(total, labels, values)
    np.maximum.at(maximum, labels, values)
    mean = total / np.maximum(count[:, None], 1)
    maximum[count == 0] = 0
    return np.concatenate((mean, maximum), -1).astype(np.float32), count


def scores(params, features, original, mass):
    h = features @ params["hidden_kernel"] + params["hidden_bias"]
    h = h * np.float32(.5) * (np.float32(1) + np.tanh(
        np.float32((2 / np.pi) ** .5) * (h + np.float32(.044715) * h ** 3)))
    residual = (h @ params["kernel"] + params["bias"])[..., 0]
    return original + np.where(mass > 0, np.float32(8) * residual, np.float32(0))


def project(head, state, base, params):
    if head.shape != base.shape or not np.isfinite(head).all():
        raise ValueError("Expected matching finite graphs")
    ids = components(base)
    valid = ids >= 0
    result = base.copy()
    if not valid.any():
        return result
    pooled, mass = descriptors(state, ids)
    original = np.bincount(ids[valid], weights=head[..., 0][valid]) / mass
    score = scores(params, pooled, original, mass)
    if not np.isfinite(score).all():
        raise ValueError("Nonfinite component scores")
    values = result[..., 0][valid]
    values[score[ids[valid]] < 0] = -20.
    result[..., 0][valid] = values
    return result
