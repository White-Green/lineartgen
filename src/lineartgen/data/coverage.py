"""Synthetic paper, filled ink and wide strokes, without super-resolution teachers."""

import numpy as np
from PIL import Image, ImageDraw

COVERAGE_KINDS = ("paper", "black", "ellipse", "polygon", "wide_stroke")


def draw_coverage(rng, size, config, *, kind=None):
    kind = kind or COVERAGE_KINDS[int(rng.integers(len(COVERAGE_KINDS)))]
    if kind not in COVERAGE_KINDS:
        raise ValueError(f"Unknown coverage kind: {kind}")
    metadata = dict(kinds=[f"coverage_{kind}"], paths=0, coverage=True, path_geometry=[])
    if kind in ("paper", "black"):
        return np.full((size, size, 1), float(kind == "paper"), np.float32), metadata
    scale = config.supersampling
    canvas = Image.new("L", (size * scale, size * scale), 255)
    draw = ImageDraw.Draw(canvas)
    center = rng.uniform(0.15, 0.85, 2) * size
    angle = rng.uniform(-np.pi, np.pi)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    if kind == "ellipse":
        radii = rng.uniform(0.15, 0.65, 2) * size
        # A dense ellipse polygon permits arbitrary rotation without image resampling.
        theta = np.linspace(0, 2 * np.pi, 257)[:-1]
        points = np.column_stack((np.cos(theta), np.sin(theta))) * radii
        points = points @ rotation.T + center
        draw.polygon([tuple(p * scale) for p in points], fill=0)
        metadata.update(center=center.tolist(), radii=radii.tolist(), angle=float(angle))
    elif kind == "polygon":
        count = int(rng.integers(3, 7))
        radii = rng.uniform(0.25, 0.80, 2) * size
        theta = np.linspace(0, 2 * np.pi, count + 1)[:-1]
        points = np.column_stack((np.cos(theta), np.sin(theta))) * radii
        points = points @ rotation.T + center
        draw.polygon([tuple(p * scale) for p in points], fill=0)
        metadata.update(vertices=points.tolist())
    else:
        length = rng.uniform(0.5, 1.6) * size
        width = float(
            np.exp(
                rng.uniform(
                    np.log(config.coverage_minimum_width), np.log(config.coverage_maximum_width)
                )
            )
        )
        a, b = np.array([[-length / 2, 0], [length / 2, 0]]) @ rotation.T + center
        normal = np.array([-np.sin(angle), np.cos(angle)]) * width / 2
        points = [a + normal, b + normal, b - normal, a - normal]
        draw.polygon([tuple(p * scale) for p in points], fill=0)
        for point in (a, b):
            lo, hi = (point - width / 2) * scale, (point + width / 2) * scale
            draw.ellipse((*lo, *hi), fill=0)
        metadata.update(endpoints=[a.tolist(), b.tolist()], width=width)
    gray = np.asarray(canvas.resize((size, size), Image.Resampling.LANCZOS), np.float32) / 255
    return gray[..., None], metadata
