"""Match normalized float inputs to a round-trip through an eight-bit image."""

import numpy as np


def quantize_source(source):
    pixels = np.rint(np.clip((np.asarray(source, np.float32) + 1) / 2, 0, 1) * 255)
    return pixels.astype(np.uint8).astype(np.float32) / 255 * 2 - 1


def quant_choice(seed, step, item):
    return bool(np.random.default_rng([seed, step, item, 0x51383030]).random() < 0.5)
