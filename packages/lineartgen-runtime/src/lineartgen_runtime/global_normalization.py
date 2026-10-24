"""Remove one opacity scale per input without changing relative stroke intensities."""

import numpy as np


def normalized_global_rough(encoded):
    """Normalize once before padding-independent area averaging and model tiling.

    `encoded` uses the existing rough-guide convention, 1 - .8 * ink.
    Each image has its own peak. The .02 floor matches the local normalizer's
    weak-signal limit. This removes absolute amplitude and depends on crop content.
    """
    value = np.asarray(encoded, np.float32)
    if value.ndim != 4 or value.shape[-1] != 1 or not np.isfinite(value).all():
        raise ValueError("Expected finite single-channel NHWC rough guide")
    ink = (1 - value) / 0.8
    peak = np.maximum(ink.max(axis=(1, 2), keepdims=True), 0.02)
    return 1 - 0.8 * ink / peak
