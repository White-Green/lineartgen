"""Exact D4 actions on projected-anchor labels, including undirected edge origins."""

import numpy as np
from learned_centerline_model import DIRECTIONS


def spatial(array, group):
    if group not in range(8):
        raise ValueError("D4 group must be 0..7")
    return np.rot90(np.flip(array, 1) if group // 4 else array, group % 4, axes=(0, 1)).copy()


def matrix(group):
    if group not in range(8):
        raise ValueError("D4 group must be 0..7")
    # Coordinates use x right, y down; numpy rot90 is counterclockwise.
    rotate = np.array([[0, 1], [-1, 0]], np.int32)
    reflect = np.diag([-1, 1]) if group // 4 else np.eye(2, dtype=np.int32)
    return np.linalg.matrix_power(rotate, group % 4) @ reflect


def points(array, size, group):
    return (np.asarray(array) - size / 2) @ matrix(group).T + size / 2


def _shift(array, dx, dy):
    n, m = array.shape
    out = np.zeros_like(array)
    out[max(0, dy) : min(n, n + dy), max(0, dx) : min(m, m + dx)] = array[
        max(0, -dy) : min(n, n - dy), max(0, -dx) : min(m, m - dx)
    ]
    return out


def target_action(truth, group):
    assert truth.ndim == 3 and truth.shape[-1] == 9 and truth.shape[0] == truth.shape[1]
    rotated = spatial(truth, group)
    out = rotated.copy()
    rotation = matrix(group)
    out[..., 1:3] = rotated[..., 1:3] @ rotation.T
    for k, direction in enumerate(DIRECTIONS):
        delta = tuple(map(int, rotation @ direction))
        if delta in DIRECTIONS:
            out[..., 4 + DIRECTIONS.index(delta)] = rotated[..., 4 + k]
        else:
            reverse = (-delta[0], -delta[1])
            out[..., 4 + DIRECTIONS.index(reverse)] = _shift(rotated[..., 4 + k], *delta)
    return out
