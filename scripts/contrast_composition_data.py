"""Independent translucent strokes with opaque geometric union targets."""

import numpy as np
from continuous_width_data import width_choice
from rough_family_data import augmentation

SINGLES = tuple(range(192)) + tuple(range(384, 432)) + tuple(range(464, 512))


def overlay_choice(seed, step, item, index):
    rng = np.random.default_rng([seed, step, item, 0x4F564552])
    enabled = bool(rng.random() < 0.5 and index in SINGLES)
    companion = int(rng.choice(SINGLES))
    return dict(
        enabled=enabled,
        index=companion,
        augmentation=augmentation(seed, step, item + 4),
        width=width_choice(seed, step, item + 4, companion),
    )


def overlay_source(first, second):
    """Black strokes on white: multiply transmittances, rather than average grays."""
    a = (np.asarray(first, np.float32) + 1) / 2
    b = (np.asarray(second, np.float32) + 1) / 2
    return (2 * a * b - 1).astype(np.float32)
