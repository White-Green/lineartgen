"""JAX counterpart of the BGRA painted-alpha contrast policy."""

import jax.numpy as jnp

from .masked_normalization import normalized_masked_guide


def normalized_painted_guide(source, noise):
    if source.shape != noise.shape or source.ndim != 4 or source.shape[-1] != 1:
        raise ValueError("Expected matching NHWC single-channel source and noise")
    maximum = jnp.max(noise, axis=(1, 2), keepdims=True)
    unpainted = noise >= maximum * (1 - 0.5 / 255)
    return normalized_masked_guide(source, unpainted.astype(noise.dtype))
