"""Normalize the original sketch before combining its cue with painted coverage."""

import numpy as np

from .masked_normalization import normalized_masked_guide


def bgra_layer_components(scribble, lineart, height, width):
    expected = height * width * 4
    if len(scribble) != expected or len(lineart) != expected:
        raise ValueError("Expected matching BGRA layer byte lengths")
    sketch = np.frombuffer(scribble, np.uint8).reshape(height, width, 4)
    painted = np.frombuffer(lineart, np.uint8).reshape(height, width, 4)
    # Match prepare_input's operation order for bit-identical full-edit cues.
    rough = (1 - sketch[..., 3].astype(np.float32) / 255) * 2 - 1
    alpha = painted[..., 3].astype(np.float32) / 255
    luma = (
        painted[..., :3].astype(np.float32) @ np.array([0.0722, 0.7152, 0.2126], np.float32) / 255
    )
    coverage = alpha * (1 - luma)
    return tuple(x[None, ..., None] for x in (rough, coverage, alpha))


def normalized_layer_guide(rough, painted_coverage, painted_alpha, mode):
    if not (
        rough.shape == painted_coverage.shape == painted_alpha.shape
        and rough.ndim == 4
        and rough.shape[-1] == 1
    ):
        raise ValueError("Expected matching NHWC single-channel layer components")
    if mode not in ("sum", "max"):
        raise ValueError("Layer combination must be sum or max")
    guide = normalized_masked_guide(rough, np.ones_like(rough))
    remaining = (1 - guide) / 0.8 * (1 - painted_alpha)
    value = (
        painted_coverage + remaining if mode == "sum" else np.maximum(painted_coverage, remaining)
    )
    # Preserve the existing full-edit input bit for bit, including at halo pixels.
    return np.where(painted_alpha == 0, guide, 1 - 0.8 * value)
