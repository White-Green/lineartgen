"""Opaque continuous capsules and rectangles, integrated on a fixed subpixel grid.

No learned opacity: gray is the coverage of a geometry boundary. The narrow linear
transition is a differentiable approximation within one subpixel, not a Gaussian
stroke profile. Fine-resolution coverage must be box-reduced for coarse stages.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


@partial(jax.jit, static_argnames=("size", "supersampling"))
def render(segments, widths, rectangles, *, size, supersampling=4):
    s = supersampling
    yy, xx = jnp.meshgrid(
        (jnp.arange(size * s) + 0.5) / s, (jnp.arange(size * s) + 0.5) / s, indexing="ij"
    )
    grid = jnp.stack((xx, yy), -1)
    alpha = jnp.zeros(grid.shape[:2], dtype=segments.dtype)

    def stroke(current, inputs):
        ends, width = inputs
        v = ends[1] - ends[0]
        t = jnp.clip(((grid - ends[0]) * v).sum(-1) / jnp.maximum((v * v).sum(), 1e-12), 0, 1)
        closest = ends[0] + t[..., None] * v
        distance = jnp.sqrt(((grid - closest) ** 2).sum(-1) + 1e-12)
        radius = 0.5 * (width[0] + t * (width[1] - width[0]))
        coverage = jnp.clip(0.5 - s * (distance - radius), 0, 1)
        coverage = jnp.where(radius > 0, coverage, 0)
        return jnp.maximum(current, coverage), None

    alpha, _ = jax.lax.scan(stroke, alpha, (segments, widths))

    def fill(current, rect):
        d = jnp.maximum(
            jnp.maximum(rect[0] - xx, xx - rect[2]), jnp.maximum(rect[1] - yy, yy - rect[3])
        )
        return jnp.maximum(current, jnp.clip(0.5 - s * d, 0, 1)), None

    alpha, _ = jax.lax.scan(fill, alpha, rectangles)
    alpha = alpha.reshape(size, s, size, s).mean((1, 3))
    return (1 - 2 * alpha)[..., None]


def geometry(metadata, *, kind=None, size=256, peak_width=2.0, margin=32):
    paths = metadata.get("path_geometry", [])
    segments, widths = [], []
    peak = max((max(p["widths"]) for p in paths), default=1)
    for path in paths:
        points = np.asarray(path["points"], np.float32) + margin
        ww = np.asarray(path["widths"], np.float32) * (peak_width / peak)
        segments.extend(np.stack((points[:-1], points[1:]), 1))
        widths.extend(np.stack((ww[:-1], ww[1:]), 1))
    rectangles = [[margin, margin, size - margin, size - margin]] if kind == "black" else []
    return (
        np.asarray(segments, np.float32).reshape(-1, 2, 2),
        np.asarray(widths, np.float32).reshape(-1, 2),
        np.asarray(rectangles, np.float32).reshape(-1, 4),
    )


def compose(rendered, source, strengths):
    """Keep protected source values exact; do not alpha-blend learned opacity."""
    return jnp.where(strengths > 0, rendered, source)
