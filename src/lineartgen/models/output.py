"""Parameter-free output choices; every head still returns a clean-image estimate."""

import jax.numpy as jnp


def denoised_output(raw, image, noise_level, mode):
    if mode == "residual":
        raw = image + jnp.sqrt(noise_level) * raw
    return jnp.clip(raw, -1, 1)
