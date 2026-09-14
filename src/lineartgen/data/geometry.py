"""Procedural clean targets, independent of the fixed evaluation shape generator.

Continuous lengths, angles, Bezier controls, offsets, widths and gaps are sampled
from optimizer-step/item keys. No evaluation image or probe is used for training.
"""

import numpy as np
from PIL import Image, ImageDraw

KINDS = ("straight", "thin", "bezier", "corner", "junction", "crossing", "parallel", "gap", "taper")


def normals(points):
    tangent = np.gradient(points, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    return np.column_stack((-tangent[:, 1], tangent[:, 0]))


def draw_geometry(rng, size, config, *, kind=None):
    """Return white=1 float32 raster and generation metadata; coordinates are subpixel."""
    if kind is None and rng.random() < config.blank_probability:
        return np.ones((size, size, 1), np.float32), {"kinds": ["blank"], "paths": 0}
    scale = config.supersampling
    canvas = Image.new("L", (size * scale, size * scale), 255)
    draw = ImageDraw.Draw(canvas)
    kinds, path_count = [], 0
    groups = 1 if kind is not None else int(rng.integers(1, 5))
    for _ in range(groups):
        selected = kind or KINDS[int(rng.integers(len(KINDS)))]
        if selected not in KINDS:
            raise ValueError(selected)
        kinds.append(selected)
        length = rng.uniform(0.20, 0.85) * size
        width = np.exp(rng.uniform(np.log(config.minimum_width), np.log(config.maximum_width)))
        if selected == "thin":
            width = rng.uniform(
                config.minimum_width, min(config.maximum_width, max(config.minimum_width, 1.8))
            )
        angle = rng.uniform(-np.pi, np.pi)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        center = rng.uniform(0.2, 0.8, 2) * size
        t = np.linspace(0, 1, 129)
        x = (t - 0.5) * length
        paths = []

        def add(points, widths=width, rotation=rotation, center=center, paths=paths):
            points = np.asarray(points, np.float64) @ rotation.T + center
            paths.append((points, np.broadcast_to(np.asarray(widths), len(points))))

        def bezier(length=length, t=t):
            controls = np.array(
                [
                    [-length / 2, 0],
                    [-length * 0.18, rng.uniform(-0.45, 0.45) * length],
                    [length * 0.18, rng.uniform(-0.45, 0.45) * length],
                    [length / 2, 0],
                ]
            )
            weights = np.stack(
                ((1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t**2, t**3), axis=1
            )
            return weights @ controls

        if selected in ("straight", "thin"):
            add(np.column_stack((x, np.zeros_like(x))))
        elif selected in ("bezier", "taper"):
            points = bezier()
            widths = width
            if selected == "taper":
                widths = width * (0.06 + 0.94 * np.sin(np.pi * t) ** rng.uniform(0.5, 1.8))
            add(points, widths)
        elif selected == "corner":
            turn = rng.uniform(0.35, 2.5)
            points = np.stack(
                (-np.cos(turn) * t[::-1] * length / 2, np.sin(turn) * t[::-1] * length / 2), axis=1
            )
            add(np.concatenate((points, np.column_stack((t[1:] * length / 2, np.zeros(128))))))
        elif selected in ("junction", "crossing"):
            directions = (
                [0, np.pi, rng.uniform(0.5, 2.6)]
                if selected == "junction"
                else [0, np.pi, rng.uniform(0.6, 2.5)]
            )
            if selected == "crossing":
                directions.append(directions[-1] + np.pi)
            for theta in directions:
                add(
                    t[:, None]
                    * length
                    * rng.uniform(0.3, 0.55)
                    * np.array([np.cos(theta), np.sin(theta)])
                )
        elif selected == "parallel":
            separation = width + np.exp(rng.uniform(np.log(1.5), np.log(18)))
            points = bezier()
            for sign in (-1, 1):
                add(points + normals(points) * (sign * separation / 2))
        else:  # Explicitly disconnected paths, including gaps between close endpoints.
            gap = min(length * 0.35, width + rng.uniform(1.5, 18))
            for a, b in ((-length / 2, -gap / 2), (gap / 2, length / 2)):
                add(np.column_stack((a + t * (b - a), np.zeros_like(t))))

        for points, widths in paths:
            normal = normals(points)
            polygon = (
                np.concatenate(
                    (
                        points + normal * widths[:, None] / 2,
                        (points - normal * widths[:, None] / 2)[::-1],
                    )
                )
                * scale
            )
            draw.polygon([tuple(p) for p in polygon], fill=0)
            path_count += 1
    gray = (
        np.asarray(canvas.resize((size, size), Image.Resampling.LANCZOS), np.float32)[..., None]
        / 255
    )
    return gray, {"kinds": kinds, "paths": path_count}


def mix_geometry(clean, config, *, seed, step, image_scale):
    """Replace selected targets; leave the original crop stream and all other RNGs untouched."""
    if config.probability == 0:
        return clean, []
    if clean.ndim != 4 or clean.shape[1] != clean.shape[2] or clean.shape[3] != 1:
        raise ValueError("Geometry replacement requires square NHWC grayscale batches")
    choose = np.random.default_rng(np.random.SeedSequence([seed, step, 0x47454F4D]))
    selected = np.flatnonzero(choose.random(len(clean)) < config.probability)
    result, records = clean.copy(), []
    for item in selected:
        key = [seed, step, int(item), 0x50415448]
        gray, metadata = draw_geometry(
            np.random.default_rng(np.random.SeedSequence(key)), clean.shape[1], config
        )
        result[item] = np.clip((gray * 2 - 1) * image_scale, -1, 1)
        records.append(dict(item=int(item), key=key, **metadata))
    return result, records
