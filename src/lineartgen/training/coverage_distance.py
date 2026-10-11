"""Coverage-preserving signed margin with Euclidean distance extensions.

Inside the antialias band the field is exactly half the physical grayscale value.
Outside, distances between pixel centers supply additional geometric supervision.
This is not an exact continuous signed distance or a guarantee of line geometry.
CPU target construction only; the renderer uses elementary JAX operations.
"""

import jax.numpy as jnp
import numpy as np
from scipy.ndimage import distance_transform_edt


def _validate(clean, radius):
    x = np.asarray(clean, dtype=np.float32)
    if x.ndim != 3 or x.shape[-1] != 1 or min(x.shape[:2]) < 1:
        raise ValueError("Expected a nonempty HWC grayscale image")
    if not np.isfinite(x).all() or np.any(np.abs(x) > 1):
        raise ValueError("Clean image must be finite and within [-1,1]")
    if not np.isfinite(radius) or radius < 1:
        raise ValueError("Radius must be finite and at least one pixel")
    return x[..., 0]


def _extension(mask, radius):
    if mask.all():
        return np.full(mask.shape, radius, np.float32)
    return np.minimum(distance_transform_edt(mask) - 0.5, radius).astype(np.float32)


def coverage_distance(clean, radius=8):
    """Preserve every fractional coverage; extend saturated regions by distance."""
    x = _validate(clean, radius)
    paper, ink = x == 1, x == -1
    field = np.where(paper, _extension(paper, radius), x * 0.5)
    field = np.where(ink, -_extension(ink, radius), field)
    return field[..., None].astype(np.float32)


def binary_distance(clean, radius=8):
    """Comparison: half-pixel-offset distance from a thresholded raster mask."""
    x = _validate(clean, radius)
    ink = x <= 0
    field = np.where(ink, -_extension(ink, radius), _extension(~ink, radius))
    return field[..., None].astype(np.float32)


def render_distance(field, source=None, strength=None):
    """A finite-width linear antialias band; optionally restore protected source."""
    prediction = jnp.clip(2 * field, -1, 1)
    if (source is None) != (strength is None):
        raise ValueError("Source and strength must be supplied together")
    if source is not None:
        prediction = jnp.where(strength > 0, prediction, source)
    return prediction
