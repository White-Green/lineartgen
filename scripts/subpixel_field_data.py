"""Sample the signed geometry of opaque capsule unions on a subpixel grid."""

from functools import partial

import jax
import jax.numpy as jnp
from subpixel_field_model import SUBPIXELS

PAD = 4


@partial(jax.jit, static_argnames=("size",))
def geometry_field(segments, widths, rectangles, *, size):
    q = (jnp.arange((size + 2 * PAD) * SUBPIXELS) + 0.5) / SUBPIXELS - PAD
    yy, xx = jnp.meshgrid(q, q, indexing="ij")
    grid = jnp.stack((xx, yy), -1)
    field = jnp.full(xx.shape, 2.0 * size, dtype=segments.dtype)

    def stroke(current, inputs):
        ends, width = inputs
        v = ends[1] - ends[0]
        t = jnp.clip(((grid - ends[0]) * v).sum(-1) / jnp.maximum((v * v).sum(), 1e-12), 0, 1)
        closest = ends[0] + t[..., None] * v
        radius = 0.5 * (width[0] + t * (width[1] - width[0]))
        d = jnp.sqrt(((grid - closest) ** 2).sum(-1) + 1e-12) - radius
        d = jnp.where(radius > 0, d, 2.0 * size)
        return jnp.minimum(current, d), None

    field, _ = jax.lax.scan(stroke, field, (segments, widths))

    def fill(current, rect):
        d = jnp.maximum(
            jnp.maximum(rect[0] - xx, xx - rect[2]), jnp.maximum(rect[1] - yy, yy - rect[3])
        )
        return jnp.minimum(current, d), None

    field, _ = jax.lax.scan(fill, field, rectangles)
    return field


def candidate_fields(base, directions, depth):
    """Global translated geometry choices sampled in physical coordinates, not resized SDFs."""
    base = base.astype(jnp.float32)
    offsets = jnp.linspace(-2, 2, 9)
    b, padded_h, padded_w = base.shape
    h, w = padded_h - 2 * PAD * SUBPIXELS, padded_w - 2 * PAD * SUBPIXELS
    results = []
    for level in range(depth):
        scale = 2**level
        y, x = jnp.meshgrid(
            (jnp.arange(h // scale) + 0.5) * scale - 0.5,
            (jnp.arange(w // scale) + 0.5) * scale - 0.5,
            indexing="ij",
        )

        def example(field, direction, x=x, y=y, scale=scale):
            def shifted(offset):
                coords = jnp.stack(
                    (
                        y + PAD * SUBPIXELS - SUBPIXELS * offset * direction[1],
                        x + PAD * SUBPIXELS - SUBPIXELS * offset * direction[0],
                    )
                )
                return (
                    jax.scipy.ndimage.map_coordinates(
                        field, coords, order=1, mode="constant", cval=2.0 * h / SUBPIXELS
                    )
                    / scale
                )

            return jax.vmap(shifted)(offsets)

        results.append(jax.vmap(example)(base, directions)[..., None])
    return tuple(results)


def central_field(base):
    p = PAD * SUBPIXELS
    return base[:, p:-p, p:-p]
