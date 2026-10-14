"""Pixel area of an opaque zero sublevel set, using piecewise-linear triangles.

Unlike a fixed transfer curve applied to field magnitude, a constant positive
field is entirely white and a constant negative field is entirely black.
Gray can only arise from triangles intersecting a sign boundary.
"""

import jax
import jax.numpy as jnp
from subpixel_field_model import SUBPIXELS


def triangle_fraction(a, b, c):
    lo = jnp.minimum(jnp.minimum(a, b), c)
    hi = jnp.maximum(jnp.maximum(a, b), c)
    mid = a + b + c - lo - hi
    one = (-lo / jnp.maximum(mid - lo, 1e-12)) * (-lo / jnp.maximum(hi - lo, 1e-12))
    two = 1 - (hi / jnp.maximum(hi - lo, 1e-12)) * (hi / jnp.maximum(hi - mid, 1e-12))
    fraction = jnp.where(mid >= 0, one, two)
    return jnp.where(hi <= 0, 1.0, jnp.where(lo >= 0, 0.0, jnp.clip(fraction, 0, 1)))


def half_grid(field, axis):
    """Retain original sample centers and insert interpolated cell/pixel boundaries."""
    pads = [(0, 0)] * field.ndim
    pads[axis] = (1, 1)
    padded = jnp.pad(field, pads, mode="edge")
    left = [slice(None)] * field.ndim
    right = left.copy()
    left[axis] = slice(None, -1)
    right[axis] = slice(1, None)
    boundaries = (padded[tuple(left)] + padded[tuple(right)]) * 0.5
    shape = list(field.shape)
    shape[axis] = field.shape[axis] * 2 + 1
    out = jnp.empty(shape, field.dtype)
    centers = [slice(None)] * field.ndim
    edges = centers.copy()
    centers[axis] = slice(1, None, 2)
    edges[axis] = slice(0, None, 2)
    return out.at[tuple(centers)].set(field).at[tuple(edges)].set(boundaries)


@jax.jit
def contour_image(field):
    b, h, w, c = field.shape
    grid = half_grid(half_grid(field, 1), 2)
    a, b_, c_, d = grid[:, :-1, :-1], grid[:, 1:, :-1], grid[:, :-1, 1:], grid[:, 1:, 1:]
    alpha = 0.5 * (triangle_fraction(a, b_, d) + triangle_fraction(a, c_, d))
    rate = 2 * SUBPIXELS
    alpha = alpha.reshape(b, h // SUBPIXELS, rate, w // SUBPIXELS, rate, c).mean((2, 4))
    return 1 - 2 * alpha
