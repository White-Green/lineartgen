"""Place training rectangles on source strokes without changing their pixel area."""

import jax
import jax.numpy as jnp


def rectangle_candidates(rectangle, key, count):
    """Integer bounds [top, left, bottom, right]; candidate zero is the input."""
    batch, height, width, _ = rectangle.shape
    rows, cols = rectangle[..., 0].any(2), rectangle[..., 0].any(1)
    top, left = rows.argmax(1), cols.argmax(1)
    h, w = rows.sum(1), cols.sum(1)
    draw = jax.random.uniform(jax.random.fold_in(key, 0x4D41534B), (batch, count - 1, 2))
    ys = (draw[..., 0] * (height - h[:, None] + 1)).astype(jnp.int32)
    xs = (draw[..., 1] * (width - w[:, None] + 1)).astype(jnp.int32)
    ys = jnp.concatenate((top[:, None], ys), 1)
    xs = jnp.concatenate((left[:, None], xs), 1)
    return jnp.stack((ys, xs, ys + h[:, None], xs + w[:, None]), -1)


def rectangle_crossing_scores(source, bounds):
    """Sum shared ink across adjacent pixels on opposite sides of each edge.

    Use only the (possibly degraded) source. Frame edges contribute zero. Prefix
    sums avoid materializing a B x candidates x H x W tensor. Scores are sampling
    weights, not a guarantee of topological continuity.
    """
    ink = jnp.clip((1 - source[..., 0]) / 2, 0, 1)
    horizontal = jnp.pad(jnp.minimum(ink[:, :-1], ink[:, 1:]), ((0, 0), (1, 1), (0, 0)))
    vertical = jnp.pad(jnp.minimum(ink[:, :, :-1], ink[:, :, 1:]), ((0, 0), (0, 0), (1, 1)))
    horizontal = jnp.pad(jnp.cumsum(horizontal, axis=2), ((0, 0), (0, 0), (1, 0)))
    vertical = jnp.pad(jnp.cumsum(vertical, axis=1), ((0, 0), (1, 0), (0, 0)))
    y0, x0, y1, x1 = (bounds[..., i] for i in range(4))
    b = jnp.arange(source.shape[0])[:, None]
    return (
        horizontal[b, y0, x1]
        - horizontal[b, y0, x0]
        + horizontal[b, y1, x1]
        - horizontal[b, y1, x0]
        + vertical[b, y1, x0]
        - vertical[b, y0, x0]
        + vertical[b, y1, x1]
        - vertical[b, y0, x1]
    )


def boundary_rectangle(source, rectangle, key, count):
    bounds = rectangle_candidates(rectangle, key, count)
    scores = rectangle_crossing_scores(source, bounds)
    chosen = bounds[jnp.arange(source.shape[0]), scores.argmax(1)]
    ys = jnp.arange(source.shape[1])[None, :, None, None]
    xs = jnp.arange(source.shape[2])[None, None, :, None]
    y0, x0, y1, x1 = (chosen[:, i, None, None, None] for i in range(4))
    return (ys >= y0) & (ys < y1) & (xs >= x0) & (xs < x1)
