"""A small ambiguity benchmark: one rough source, many acceptable clean strokes."""

import hashlib

import numpy as np
from cue_preserving_data import probes_for
from PIL import Image, ImageDraw
from scipy import ndimage

from lineartgen.config import GeometryDataConfig
from lineartgen.data.geometry import KINDS, draw_geometry, normals

SIZE = 256
COUNT = 64
SEED = 2026101301
OPACITY = 0.4
OFFSETS = np.arange(-2, 2.01, 0.5)
WIDTHS = (0.8, 1.0, 1.2)
CHOICES = tuple((float(offset), width) for offset in OFFSETS for width in WIDTHS)
ENDPOINTS = (1, 25)
CENTER = 13


def raster(metadata, direction, offset, width_factor, *, size=SIZE):
    scale = 4
    canvas = Image.new("L", (size * scale, size * scale), 255)
    draw = ImageDraw.Draw(canvas)
    for path in metadata.get("path_geometry", []):
        points = np.asarray(path["points"]) + 32 + offset * direction
        widths = np.asarray(path["widths"]) * width_factor
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
        draw.polygon([tuple(x) for x in polygon], fill=0)
    return (
        np.asarray(canvas.resize((size, size), Image.Resampling.LANCZOS), np.float32) / 127.5 - 1
    )[..., None]


def profile_samples(image, probes):
    offset = np.linspace(-1, 1, 129)[None] * probes[:, 4:5]
    x = probes[:, 0:1] + probes[:, 2:3] * offset
    y = probes[:, 1:2] + probes[:, 3:4] * offset
    alpha = np.clip((1 - image[..., 0]) / 2, 0, 1)
    return ndimage.map_coordinates(alpha, [y, x], order=1, mode="constant", cval=0)


def make_case(split, index):
    assert split in ("train", "probe") and 0 <= index < COUNT
    rng = np.random.default_rng([SEED, int(split == "probe"), index])
    cfg = GeometryDataConfig(minimum_width=0.8, maximum_width=5, blank_probability=0)
    kind = (
        KINDS[index // 6]
        if index < 54
        else "blank"
        if index < 56
        else "black"
        if index < 58
        else "wide"
    )
    if kind in ("blank", "black"):
        image = np.ones((SIZE, SIZE, 1), np.float32)
        if kind == "black":
            image[32:-32, 32:-32] = -1
        variants = np.stack([image] * len(CHOICES))
        metadata = dict(kinds=[kind], paths=0)
        direction = np.array([1.0, 0.0])
    else:
        if kind == "wide":
            cfg = cfg.model_copy(update={"minimum_width": 16, "maximum_width": 32})
        _, metadata = draw_geometry(
            rng, SIZE - 64, cfg, kind="straight" if kind == "wide" else kind, include_paths=True
        )
        direction = normals(np.asarray(metadata["path_geometry"][0]["points"]))[64]
        variants = np.stack([raster(metadata, direction, o, w) for o, w in CHOICES])
    mixture = variants[list(ENDPOINTS)].mean(0)
    source = 1 - 2 * OPACITY * ndimage.gaussian_filter(
        (1 - mixture) / 2, (1.2, 1.2, 0), mode="constant", cval=0
    )
    probes, connections = probes_for(metadata, margin=32)
    if len(probes):
        probes[:, 4] += 4
    profiles = np.stack([profile_samples(y, probes) for y in variants])
    data = dict(
        variants=variants,
        source=source.astype(np.float32),
        mixture=mixture,
        probes=probes,
        connections=connections,
        profiles=profiles,
        direction=direction,
    )
    record = dict(
        split=split,
        index=index,
        kind=kind,
        direction=direction.tolist(),
        metadata=metadata,
        variants_sha256=hashlib.sha256(variants.tobytes()).hexdigest(),
        source_sha256=hashlib.sha256(data["source"].tobytes()).hexdigest(),
    )
    return data, record


def shape_metrics(prediction, data, strength):
    """Profile distance to an allowed family, not to a designated paired teacher.

    Different probes may choose different allowed offsets/widths. Continuity is
    evaluated separately. This finite family is a toy diagnostic, not a general
    human preference model or a proof that other clean drawings are invalid.
    """
    alpha = np.clip((1 - prediction[..., 0]) / 2, 0, 1)
    active = strength[..., 0] > 0
    allowed = np.max((1 - data["variants"]) / 2, axis=0)[..., 0] > 0.02
    outside = active & ~ndimage.binary_dilation(allowed, iterations=1)
    result = dict(
        off_family_ink_fraction=float(alpha[outside].sum() / max(alpha[active].sum(), 1)),
        off_family_mean_opacity=float(alpha[outside].mean()) if outside.any() else None,
    )
    probes = data["probes"]
    if not len(probes):
        return result
    # Exclude profiles touching the protected area; retain/report all whole-image profiles.
    offsets = np.linspace(-1, 1, 129)[None] * probes[:, 4:5]
    xs = probes[:, 0:1] + probes[:, 2:3] * offsets
    ys = probes[:, 1:2] + probes[:, 3:4] * offsets
    visible = np.all(
        ndimage.map_coordinates(active.astype(float), [ys, xs], order=0, mode="constant", cval=0)
        > 0.5,
        axis=1,
    )
    obs = profile_samples(prediction, probes)
    reference = data["profiles"]
    masses = np.trapezoid(reference, offsets[None], axis=-1)
    errors = np.trapezoid(abs(reference - obs[None]), offsets[None], axis=-1) / np.maximum(
        masses, 0.25
    )
    observed_mass = np.trapezoid(obs, offsets, axis=-1)
    result |= dict(
        profiles_valid=int(visible.sum()),
        profiles_total=len(probes),
        family_profile_error=float(errors[:, visible].min(0).mean()) if visible.any() else None,
        missing_profile_fraction=float(
            (observed_mass[visible] < 0.25 * masses[:, visible].min(0)).mean()
        )
        if visible.any()
        else None,
    )
    return result
