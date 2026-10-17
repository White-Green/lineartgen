"""Two coherent thin interpretations of an isolated near-square broad source."""

import numpy as np
from procedural_shape_data import PAD, SIZE, SUBPIXELS
from procedural_shape_data import render as render_primary

SEED = 2026101706


def selected(step):
    # A possible later750-batch addition, disjoint from every GL batch.
    return step // 6 % 8 == 5


def record(step, item):
    rng = np.random.default_rng([SEED, step, item])
    square = bool(rng.integers(4) == 0)
    width = rng.uniform(16, 88)
    ratio = 1.0 if square else np.exp(rng.uniform(np.log(0.75), np.log(4 / 3)))
    length = width / ratio
    kind = int(rng.integers(4))
    axis = int(rng.integers(4)) * np.pi / 2
    angle = (
        axis
        if kind == 0
        else axis + rng.uniform(-np.pi / 60, np.pi / 60)
        if kind == 1
        else rng.uniform(-np.pi, np.pi)
    )
    c, s = abs(np.cos(angle)), abs(np.sin(angle))
    rx, ry = (c * width + s * length) / 2, (s * width + c * length) / 2
    cx, cy = rng.uniform(8 + rx, 248 - rx), rng.uniform(8 + ry, 248 - ry)
    canonical_style = bool(rng.integers(4) == 0)
    sigma, opacity = (
        (1.2, 0.4) if canonical_style else (rng.uniform(0.6, 2.4), rng.uniform(0.2, 0.8))
    )
    return dict(
        step=step,
        item=item,
        depth=step % 6 + 1,
        width=float(width),
        length=float(length),
        angle=float(angle),
        center=[float(cx), float(cy)],
        sigma=float(sigma),
        opacity=float(opacity),
        angle_kind=kind,
        canonical_style=canonical_style,
        square=square,
        filled=False,
    )


def validate(r):
    w, h, theta = r["width"], r["length"], r["angle"]
    c, s = abs(np.cos(theta)), abs(np.sin(theta))
    rx, ry = (c * w + s * h) / 2, (s * w + c * h) / 2
    cx, cy = r["center"]
    assert 16 <= w <= 88 and 12 <= h <= 88 * 4 / 3
    assert 0.75 - 1e-12 <= w / h <= 4 / 3 + 1e-12
    assert 8 <= cx - rx and cx + rx <= 248
    assert 8 <= cy - ry and cy + ry <= 248
    assert np.isfinite([w, h, theta, cx, cy, r["sigma"], r["opacity"]]).all()
    assert 0.6 <= r["sigma"] <= 2.4 and 0.2 <= r["opacity"] <= 0.8
    assert not r["filled"]
    if r["square"]:
        assert w == h


def render(r):
    result = render_primary(r)
    q = (np.arange((SIZE + 2 * PAD) * SUBPIXELS, dtype=np.float64) + 0.5) / SUBPIXELS - PAD
    yy, xx = np.meshgrid(q, q, indexing="ij")
    c, s = np.cos(r["angle"]), np.sin(r["angle"])
    x, y = xx - r["center"][0], yy - r["center"][1]
    u, v = c * x - s * y, s * x + c * y
    axis_base = np.maximum(abs(u) - r["width"] / 2, abs(v) - 1).astype(np.float16)
    p = PAD * SUBPIXELS
    alpha = np.clip(0.5 - SUBPIXELS * axis_base[p:-p, p:-p].astype(np.float32), 0, 1)
    axis_clean = 1 - 2 * alpha.reshape(SIZE, SUBPIXELS, SIZE, SUBPIXELS).mean((1, 3))
    return result | dict(
        axis_base=axis_base,
        axis_direction=np.array([s, c], np.float32),
        axis_clean=axis_clean[..., None],
    )


def permitted_outputs(payload, strength):
    """No perpendicular escape from the active region in partial application."""
    allowed = bool(np.all(strength > 0))
    original = np.where(strength > 0, payload["clean"], payload["source"])
    other = payload["axis_clean"] if allowed else payload["clean"]
    return original, np.where(strength > 0, other, payload["source"]), allowed
