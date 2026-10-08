"""Parameter-free output choices; every head still returns a clean-image estimate."""

import jax
import jax.numpy as jnp


@jax.custom_jvp
def straight_through_clip(value):
    """Exact clipped forward with an identity surrogate derivative, not its true derivative."""
    return jnp.clip(value, -1, 1)


@straight_through_clip.defjvp
def _straight_through_clip_jvp(primals, tangents):
    (value,), (tangent,) = primals, tangents
    return straight_through_clip(value), tangent


def denoised_output(raw, image, noise_level, mode, clip_gradient="hard", *, return_preclip=False):
    if mode == "residual":
        raw = image + jnp.sqrt(noise_level) * raw
    if clip_gradient == "straight_through":
        prediction = straight_through_clip(raw)
    elif clip_gradient == "hard":
        prediction = jnp.clip(raw, -1, 1)
    else:
        raise ValueError(f"Unknown output clip gradient: {clip_gradient}")
    # The optional value is the physical clean estimate, not the residual head.
    return (prediction, raw) if return_preclip else prediction
