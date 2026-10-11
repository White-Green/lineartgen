"""Direct supervision of a preclip field by coverage-preserving distance targets."""

import jax.numpy as jnp

from .interval import interval_absolute_error
from .losses import tone_weights


def distance_supervision(preclip, distance, clean, active, radius=8):
    """Regress normalized distance; permit saturation beyond the truncated cap.

    Rendered prediction remains clip(preclip,-1,1). This treats preclip/2 as the
    learned margin field. No gradient is taken through a predicted EDT or a hard
    threshold. Target distance is a CPU-prepared constant. Tone balance matches
    the existing power.5 image objective; the term has no noise-level division.
    """
    error = interval_absolute_error(preclip / (2 * radius), distance / radius)
    weights = tone_weights(clean, 0.5) * active
    per_image = (error * weights).sum((1, 2, 3)) / jnp.maximum(weights.sum((1, 2, 3)), 1e-6)
    return per_image.mean()
