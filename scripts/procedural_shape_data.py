"""Deterministic, varied rectangle/bar recipes and their physical target fields."""

import numpy as np
from scipy.ndimage import gaussian_filter

SEED = 2026101702
SIZE = 256
PAD = 4
SUBPIXELS = 4


def selected(step):
    return step // 6 % 4 == 3


def record(step, item):
    rng = np.random.default_rng([SEED, step, item])
    filled = bool(item % 2)
    rejected = 0
    while True:
        if filled:
            width, length = rng.uniform(128, 208, 2)
            if not 0.8 <= width / length <= 1.25:
                rejected += 1
                continue
        else:
            width = rng.uniform(24, 88)
            length = rng.uniform(max(128, 2 * width), 208)
        kind = int(rng.integers(4))
        axis = int(rng.integers(4)) * np.pi / 2
        angle = (
            axis
            if kind == 0
            else (
                axis + rng.uniform(-np.pi / 60, np.pi / 60)
                if kind == 1
                else rng.uniform(-np.pi, np.pi)
            )
        )
        c, s = abs(np.cos(angle)), abs(np.sin(angle))
        rx, ry = (c * width + s * length) / 2, (s * width + c * length) / 2
        if max(rx, ry) <= 120:
            break
        rejected += 1
    cx, cy = rng.uniform(8 + rx, 248 - rx), rng.uniform(8 + ry, 248 - ry)
    canonical_style = bool(rng.integers(4) == 0)
    sigma, opacity = (
        (1.2, 0.4) if canonical_style else (rng.uniform(0.6, 2.4), rng.uniform(0.2, 0.8))
    )
    return dict(
        step=step,
        item=item,
        depth=step % 6 + 1,
        filled=filled,
        width=float(width),
        length=float(length),
        angle=float(angle),
        center=[float(cx), float(cy)],
        sigma=float(sigma),
        opacity=float(opacity),
        angle_kind=kind,
        canonical_style=canonical_style,
        rejected=rejected,
    )


def validate(r):
    w, h, theta = r["width"], r["length"], r["angle"]
    c, s = abs(np.cos(theta)), abs(np.sin(theta))
    rx, ry = (c * w + s * h) / 2, (s * w + c * h) / 2
    cx, cy = r["center"]
    assert 8 - 1e-10 <= cx - rx and cx + rx <= 248 + 1e-10
    assert 8 - 1e-10 <= cy - ry and cy + ry <= 248 + 1e-10
    assert np.isfinite([w, h, theta, cx, cy, r["sigma"], r["opacity"]]).all()
    assert 0.6 <= r["sigma"] <= 2.4 and 0.2 <= r["opacity"] <= 0.8
    if r["filled"]:
        assert 128 <= w <= 208 and 128 <= h <= 208 and 0.8 <= w / h <= 1.25
    else:
        assert 24 <= w <= 88 and max(128, 2 * w) <= h <= 208


def render(r):
    q = (np.arange((SIZE + 2 * PAD) * SUBPIXELS, dtype=np.float64) + 0.5) / SUBPIXELS - PAD
    yy, xx = np.meshgrid(q, q, indexing="ij")
    theta = r["angle"]
    c, s = np.cos(theta), np.sin(theta)
    x, y = xx - r["center"][0], yy - r["center"][1]
    u, v = c * x - s * y, s * x + c * y
    target_width = r["width"] if r["filled"] else 2
    base = np.maximum(abs(u) - target_width / 2, abs(v) - r["length"] / 2).astype(np.float16)
    p = PAD * SUBPIXELS
    source_alpha = (
        (abs(u[p:-p, p:-p]) < r["width"] / 2) & (abs(v[p:-p, p:-p]) < r["length"] / 2)
    ).astype(np.float32)
    source_alpha = source_alpha.reshape(SIZE, SUBPIXELS, SIZE, SUBPIXELS).mean((1, 3))
    source = 1 - 2 * r["opacity"] * gaussian_filter(source_alpha, r["sigma"], mode="constant")
    alpha = np.clip(0.5 - SUBPIXELS * base[p:-p, p:-p].astype(np.float32), 0, 1)
    clean = 1 - 2 * alpha.reshape(SIZE, SUBPIXELS, SIZE, SUBPIXELS).mean((1, 3))
    direction = np.zeros(2, np.float32) if r["filled"] else np.array([c, -s], np.float32)
    assert source.shape == clean.shape == (SIZE, SIZE)
    assert source.dtype == clean.dtype == np.float32 and np.isfinite(base).all()
    return dict(source=source[..., None], clean=clean[..., None], base=base, direction=direction)
