"""Independent noise streams at every spatial resolution of the latent sampler."""

from typing import NamedTuple

import numpy as np


class StageNoise(NamedTuple):
    observation: np.ndarray
    initial: np.ndarray
    reverse: np.ndarray


def stage_noise(seed, level, shape, factor, channels, steps, *, stochastic=True):
    batch, h, w, _ = shape
    if min(h, w) < 64:
        latent = shape
    else:
        latent = (batch, (h + factor - 1) // factor, (w + factor - 1) // factor, channels)

    def normal(stream, size):
        sequence = np.random.SeedSequence([int(seed) % (1 << 64), level, stream])
        return np.random.Generator(np.random.PCG64(sequence)).standard_normal(
            size, dtype=np.float32
        )

    # The last step is deterministic. Its zero noise is supplied by the sampler.
    reverse = np.empty((steps - 1 if stochastic else 0, *latent), np.float32)
    for index in range(len(reverse)):
        reverse[index] = normal(index + 2, latent)
    return StageNoise(normal(0, shape), normal(1, latent), reverse)


def pyramid_noise(seed, shape, depth, factor, channels, steps):
    levels = []
    batch, h, w, c = shape
    for level in range(depth):
        levels.append(stage_noise(seed, level, (batch, h, w, c), factor, channels, steps))
        h, w = (h + 1) // 2, (w + 1) // 2
    return tuple(levels)
