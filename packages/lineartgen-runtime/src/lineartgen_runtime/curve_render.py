"""Exact opaque union of discs along a variable-width line segment."""

import numpy as np


def optimum(dx, dy, vx, vy, r0, r1):
    length2 = vx * vx + vy * vy
    length = np.sqrt(np.maximum(length2, 1e-24))
    x = (dx * vx + dy * vy) / length
    perpendicular = np.sqrt(np.maximum(dx * dx + dy * dy - x * x, 0) + 1e-12)
    slope = (r1 - r0) / length
    along = x + slope * perpendicular / np.sqrt(np.maximum(1 - slope * slope, 1e-24))
    t = np.clip(along / length, 0, 1)
    t = np.where(slope >= 1, 1, np.where(slope <= -1, 0, t))
    # Keep constant-radius operation order identical to the existing renderer.
    t = np.where(r0 == r1, np.clip((dx * vx + dy * vy) / np.maximum(length2, 1e-12), 0, 1), t)
    return np.where(length2 == 0, np.asarray(r1 > r0, np.float64), t)


def render_tile(segments, widths, origin, size, supersampling):
    x0, y0 = origin
    h, w = size
    s = supersampling
    alpha = np.zeros((h * s, w * s), np.float64)
    floor = np.array([x0, y0]) * s
    ceiling = floor + np.array([w, h]) * s
    extent = min(max(h, w) * s, int(np.ceil(widths.max() * s)) + 4 * s)
    batch_size = max(1, min(128, 2_000_000 // max(1, extent**2)))
    for start in range(0, len(segments), batch_size):
        lines = segments[start : start + batch_size]
        ww = np.asarray(widths[start : start + batch_size], np.float64)
        pad = ww.max(axis=1) / 2 + 1 / s
        low = np.maximum(np.floor((lines.min(1) - pad[:, None]) * s).astype(int), floor)
        high = np.minimum(np.ceil((lines.max(1) + pad[:, None]) * s).astype(int), ceiling)
        valid = np.all(high > low, axis=1)
        lines, ww, low, high = lines[valid], ww[valid], low[valid], high[valid]
        if not len(lines):
            continue
        width, height = (high - low).max(axis=0)
        xx = low[:, 0, None, None] + np.arange(width)[None, None, :]
        yy = low[:, 1, None, None] + np.arange(height)[None, :, None]
        dx = (xx + 0.5) / s - lines[:, 0, 0, None, None]
        dy = (yy + 0.5) / s - lines[:, 0, 1, None, None]
        v = lines[:, 1] - lines[:, 0]
        vx, vy = v[:, 0, None, None], v[:, 1, None, None]
        t = optimum(dx, dy, vx, vy, ww[:, 0, None, None] / 2, ww[:, 1, None, None] / 2)
        px = lines[:, 0, 0, None, None] + t * vx
        py = lines[:, 0, 1, None, None] + t * vy
        distance = np.sqrt(((xx + 0.5) / s - px) ** 2 + ((yy + 0.5) / s - py) ** 2 + 1e-12)
        radius = 0.5 * (ww[:, 0, None, None] + t * (ww[:, 1] - ww[:, 0])[:, None, None])
        coverage = np.where(radius > 0, np.clip(0.5 - s * (distance - radius), 0, 1), 0)
        active = (xx < high[:, 0, None, None]) & (yy < high[:, 1, None, None])
        indices = (yy - floor[1]) * (w * s) + (xx - floor[0])
        np.maximum.at(alpha.ravel(), indices[active], coverage[active])
    return (1 - 2 * alpha.reshape(h, s, w, s).mean((1, 3))).astype(np.float32)[..., None]


def render_curves(output, size=256):
    """Merge geometry and width knots without imposing uniform stroke width."""
    shape, widths, positions, flags = (np.asarray(v) for v in output)
    expected = ((1, 16, 256, 2), (1, 16, 128), (1, 16, 128), (1, 16, 2))
    if tuple(v.shape for v in (shape, widths, positions, flags)) != expected:
        raise ValueError("Invalid curve output shapes")
    if not all(np.isfinite(v).all() for v in (shape, widths, positions, flags)):
        raise RuntimeError("Curve inference produced nonfinite output")
    if np.any(widths <= 0) or np.any(np.diff(positions, axis=-1) <= 0):
        raise RuntimeError("Curve widths and width positions must be positive and ordered")
    shape, widths, positions, flags = (v[0] for v in (shape, widths, positions, flags))
    grid = np.linspace(0, 1, shape.shape[-2])
    segments, thicknesses = [], []
    for q in np.flatnonzero(flags[:, 0] >= 0):
        u = np.unique(np.r_[grid, positions[q]])
        points = np.stack([np.interp(u, grid, shape[q, :, i]) for i in range(2)], -1)
        width = np.interp(u, positions[q], widths[q])
        if flags[q, 1] >= 0:
            points[-1], width[-1] = points[0], width[0]
        segments.append(np.stack((points[:-1], points[1:]), 1) * size)
        thicknesses.append(np.stack((width[:-1], width[1:]), -1) * size)
    if not segments:
        return np.ones((size, size, 1), np.float32)
    return render_tile(
        np.concatenate(segments), np.concatenate(thicknesses), (0, 0), (size, size), 4
    )
