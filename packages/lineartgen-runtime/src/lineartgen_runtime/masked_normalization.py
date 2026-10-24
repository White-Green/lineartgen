"""Local contrast without borrowing a protected stroke's higher opacity."""

import numpy as np


def normalized_masked_guide(source, noise):
    if source.shape != noise.shape or source.ndim != 4 or source.shape[-1] != 1:
        raise ValueError("Expected matching NHWC single-channel source and noise")
    ink = np.clip((1 - source) / 2, 0, 1)
    editable = noise > 0
    peaks = np.concatenate((np.where(editable, ink, 0), np.where(editable, 0, ink)), axis=-1)
    for axis in (1, 2):
        pads = [(0, 0)] * 4
        pads[axis] = (16, 16)
        padded = np.pad(peaks, pads)
        peaks = np.zeros_like(peaks)
        for shift in range(33):
            section = [slice(None)] * 4
            section[axis] = slice(shift, shift + peaks.shape[axis])
            np.maximum(peaks, padded[tuple(section)], out=peaks)
    peak = np.where(editable, peaks[..., :1], peaks[..., 1:])
    return 1 - 0.8 * ink / np.maximum(peak, 0.02)
