"""Freeze sixteen new synthetic drawings for the final Krita preview check only."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from centerline_grid import render
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

SEED = 2026101912


def path_segments(points):
    points = np.asarray(points, np.float64)
    return np.stack((points[:-1], points[1:]), axis=1)


def curve(rng, size):
    points = np.array([[0.16, 0.72], [0.32, 0.08], [0.65, 0.91], [0.84, 0.28]]) * size
    points[1:3] += rng.uniform(-0.06, 0.06, (2, 2)) * size
    t = np.linspace(0, 1, 121)[:, None]
    return (
        (1 - t) ** 3 * points[0]
        + 3 * (1 - t) ** 2 * t * points[1]
        + 3 * (1 - t) * t * t * points[2]
        + t**3 * points[3]
    )


def make_case(index):
    rng = np.random.default_rng([SEED, index])
    size = 512 if index == 15 else 256
    paths, rectangles = [], []
    widths = []
    clean_widths = []
    kind = "curve"

    def add(points, width=6.0, clean=2.0):
        lines = path_segments(points)
        paths.append(lines)
        widths.append(np.full((len(lines), 2), width))
        clean_widths.append(np.full((len(lines), 2), clean))

    c = curve(rng, size)
    if index in (0, 1, 2, 3):
        add(c, width=7 if index != 3 else 11)
        if index == 2:
            kind = "repeated_sketch_curve"
        if index == 3:
            w = np.linspace(2, 11, len(c))
            widths[-1] = np.stack((w[:-1], w[1:]), axis=1)
            w = np.linspace(0.8, 2.4, len(c))
            clean_widths[-1] = np.stack((w[:-1], w[1:]), axis=1)
    elif index in (4, 5):
        kind = "clear_pair" if index == 4 else "ambiguous_pair"
        gap = 14 if index == 4 else 3.2
        add(c - [0, gap / 2], width=5.5)
        add(c + [0, gap / 2], width=5.5)
    elif index in (6, 7):
        kind = "crossing" if index == 6 else "junction"
        add([[43, 73], [211, 178]], width=9)
        add([[64, 207], [191, 52]] if index == 6 else [[93, 104], [50, 207]], width=6)
    elif index in (8, 9):
        kind = "explicit_gap" if index == 8 else "broken_sketch"
        add([[40, 168], [124, 131]], width=6)
        add([[133, 127], [217, 90]], width=6)
    elif index in (10, 11):
        kind = "very_wide_line" if index == 10 else "short_wide_line"
        center = rng.uniform(112, 144, 2)
        direction = np.array([0.8, -0.6])
        length = 150 if index == 10 else 62
        add(
            [center - direction * length / 2, center + direction * length / 2],
            width=72 if index == 10 else 28,
        )
    elif index in (12, 13):
        kind = "fill" if index == 12 else "fill_and_curve"
        rectangles = [[42, 58, 153, 191]] if index == 12 else [[22, 30, 100, 112]]
        if index == 13:
            add(c * 0.6 + [80, 70], width=9)
    elif index == 14:
        kind = "blank"
    elif index == 15:
        kind = "mixed_scene"
        add(c * 0.65 + [80, 0], width=9)
        add([[60, 320], [230, 410]], width=14)
        add([[240, 416], [411, 480]], width=14)
        add(c * 0.35 + [265, 140], width=4)
        add(c * 0.35 + [265, 151], width=4)
        rectangles = [[30, 28, 110, 103]]
    segments = np.concatenate(paths) if paths else np.empty((0, 2, 2))
    wide = np.concatenate(widths) if widths else np.empty((0, 2))
    thin = np.concatenate(clean_widths) if clean_widths else np.empty((0, 2))
    rects = np.asarray(rectangles).reshape(-1, 4)
    clean = render(segments, thin, rects, size)
    rough = render(segments, wide, rects, size)
    if index == 2:
        rough = np.minimum(rough, render(segments + [1.4, -1.0], wide * 0.55, rects, size))
        rough = np.minimum(rough, render(segments + [-1.2, 1.3], wide * 0.55, rects, size))
    if index == 9:
        rough[137:145] = 1
    opacity = 0.18 if index == 1 else float(rng.uniform(0.35, 0.8))
    sigma = float(rng.uniform(0.4, 1.4))
    source = (
        1
        - 2 * opacity * gaussian_filter((1 - rough) / 2, (sigma, sigma, 0), mode="constant", cval=0)
    ).astype(np.float32)
    return dict(
        source=source, teacher=clean, segments=segments, widths=thin, rectangles=rects
    ), dict(
        index=index,
        kind=kind,
        size=size,
        opacity=opacity,
        blur_sigma=sigma,
        ambiguity_allowed=index in (2, 5, 9, 11),
    )


def prepare(root, training):
    root.mkdir(parents=True, exist_ok=False)
    hashes = set()
    for p in training.glob("inputs/*.npz"):
        with np.load(p) as z:
            hashes.add(hashlib.sha256(z["source"].tobytes()).hexdigest())
    records = []
    for index in range(16):
        arrays, record = make_case(index)
        source = arrays["source"]
        assert np.isfinite(source).all() and source.min() >= -1 and source.max() <= 1
        digest = hashlib.sha256(source.tobytes()).hexdigest()
        if record["kind"] != "blank":
            assert digest not in hashes
        p = root / f"{index:02d}.npz"
        np.savez_compressed(p, **arrays)
        record.update(
            file=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest(), source_sha256=digest
        )
        records.append(record)
        size = record["size"]
        canvas = Image.new("RGB", (size * 2, size + 28), "#ddd")
        draw = ImageDraw.Draw(canvas)
        for k, name in enumerate(("source", "teacher")):
            draw.text((k * size + 3, 3), f"{index} {record['kind']} {name}", fill="black")
            pixels = np.rint((arrays[name][..., 0] + 1) * 127.5).astype(np.uint8)
            canvas.paste(Image.fromarray(pixels).convert("RGB"), (k * size, 28))
        canvas.save(root / f"{index:02d}.png")
    (root / "manifest.json").write_text(
        json.dumps(
            dict(
                seed=SEED,
                records=records,
                used_for_training=False,
                predictions_made=False,
                selection_use=False,
                purpose=(
                    "Final preview check of new synthetic drawings; "
                    "not the reserved historical DN corpus."
                ),
                recipe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument(
        "--training", type=Path, default=Path("/tmp/lineart-graph-training-cache-20261018")
    )
    prepare(**vars(p.parse_args()))
