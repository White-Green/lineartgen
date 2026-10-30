"""Native-resolution inputs for the synthetic rough-cleanup curve preview."""

from dataclasses import dataclass

import numpy as np

from .layer_normalization import bgra_layer_components, normalized_layer_guide

TILE = 256
CORE = 128
SELECTORS = np.array([1, 1, 0, 0, 1, 0], np.float32)


@dataclass(frozen=True)
class Axis:
    size: int
    core: int
    pad: int
    padded: int


def axis_layout(size):
    if size <= TILE:
        return Axis(size, size, 0, TILE)
    return Axis(size, CORE, (TILE - CORE) // 2, (size + CORE - 1) // CORE * CORE + TILE - CORE)


def downsample(image):
    batch, height, width, channels = image.shape
    return image.reshape(batch, height // 2, 2, width // 2, 2, channels).mean((2, 4))


def levels_from_bgra(scribble, lineart, width, height, strength, seed, depth):
    """Normalize once, area-downsample, then add independent noise at each level.

    The NumPy RNG is explicitly different from JAX's RNG. It has the same Gaussian
    law, and its stream is stable across tile traversal and recursion-depth changes.
    """
    rows, cols = axis_layout(height), axis_layout(width)
    rough, coverage, alpha = bgra_layer_components(scribble, lineart, height, width)
    guide = normalized_layer_guide(rough, coverage, alpha, "sum")
    base = np.zeros((1, rows.padded, cols.padded, 4), np.float32)
    base[..., 3] = 1
    base[:, rows.pad : rows.pad + height, cols.pad : cols.pad + width] = np.concatenate(
        (0.4 * (1 - rough) / 2, 0.4 * coverage, strength * (1 - alpha), guide), -1
    )
    result = []
    for level in range(depth):
        rng = np.random.default_rng(
            np.random.SeedSequence([seed & 0xFFFFFFFF, seed >> 32, level, 20261030])
        )
        noise = rng.standard_normal(base[..., :1].shape, dtype=np.float32)
        noisy = base[..., 3:4] + 0.25 * base[..., 2:3] * noise
        result.append(
            np.concatenate((base[..., :3], (1 - noisy) / 2, (1 - base[..., 3:4]) / 2), -1)
        )
        if level + 1 < depth:
            base = downsample(base)
    return tuple(result), rows, cols


def tile_inputs(levels, y, x):
    result = []
    for level, image in enumerate(levels):
        divisor = 2**level
        size = TILE // divisor
        crop = image[:, y // divisor : y // divisor + size, x // divisor : x // divisor + size]
        if crop.shape[1:3] != (size, size):
            raise ValueError("Curve tile leaves the padded input canvas")
        flags = np.broadcast_to(SELECTORS, (*crop.shape[:-1], len(SELECTORS)))
        result.append(np.concatenate((crop, flags), -1))
    return tuple(result)
