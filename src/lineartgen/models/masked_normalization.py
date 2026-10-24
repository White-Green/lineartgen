"""Training counterpart of the runtime's protection-aware local contrast."""

import jax
import jax.numpy as jnp


def normalized_masked_guide(source, noise):
    if source.shape != noise.shape or source.ndim != 4 or source.shape[-1] != 1:
        raise ValueError("Expected matching NHWC single-channel source and noise")
    ink = jnp.clip((1 - source) / 2, 0, 1)
    editable = noise > 0
    peaks = jnp.concatenate((jnp.where(editable, ink, 0), jnp.where(editable, 0, ink)), axis=-1)
    for window in ((1, 33, 1, 1), (1, 1, 33, 1)):
        peaks = jax.lax.reduce_window(peaks, 0.0, jax.lax.max, window, (1, 1, 1, 1), "SAME")
    peak = jnp.where(editable, peaks[..., :1], peaks[..., 1:])
    return 1 - 0.8 * ink / jnp.maximum(peak, 0.02)
