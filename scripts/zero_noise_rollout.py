"""Diagnostic deterministic restoration, with no added Gaussian field."""

import jax.numpy as jnp
from anchored_endpoint import velocity

from lineartgen.diffusion import pyramid


def without_noise(prepared):
    return {**prepared, "images": prepared["guides"]}


def integrate(predict, source, strength, *, depth, steps):
    if steps < 1:
        raise ValueError("steps must be positive")
    guides, levels = pyramid(source, depth), pyramid(strength, depth)
    images = guides
    for step in range(steps):
        remaining = tuple(jnp.full_like(x, 1 - step / steps) for x in images)
        estimates = predict(images, levels, guides, remaining)
        images = tuple(
            jnp.where(t > 0, x + v / steps, s)
            for x, v, t, s in zip(images, estimates, levels, guides, strict=True)
        )
    return images[0]


def run_sample(model, source, strength, key, depth, steps=32):
    del key  # The diagnostic removes only the additional Gaussian field.
    return integrate(
        lambda x, t, s, u: velocity(model, x, t, s, u, False),
        source,
        strength,
        depth=depth,
        steps=steps,
    )
