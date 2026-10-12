"""Fixed, invertible synthetic corruption for a capacity/learnability diagnostic."""

import hashlib

import numpy as np

from lineartgen.config import GeometryDataConfig
from lineartgen.data.geometry import KINDS, draw_geometry, normals

SIZE = 256
COUNT = 96
SEED = 2026101291
CORE = 0.4
OPACITY = 0.4
SIGMA = 1.5
MODES = ("whole01", "whole05", "partial05")


def transfer(shape, core=CORE):
    h, w = shape

    def kernel(n):
        distance = np.minimum(np.arange(n), n - np.arange(n))
        weights = np.exp(-distance**2 / (2 * SIGMA**2))
        return weights / weights.sum()

    # A positive, sampled Gaussian avoids tiny negative lobes from truncating
    # the continuous Fourier Gaussian at Nyquist. No source clipping is needed.
    gaussian = np.fft.fft(kernel(h)).real[:, None] * np.fft.rfft(kernel(w)).real[None, :]
    return core + (1 - core) * gaussian


def corrupt(clean, core=CORE):
    """Retain a core plus Gaussian-blurred ink, then attenuate its opacity.

    The fixed periodic linear operator has spectrum in [core,1]. Training
    teachers have a 24px white margin; this is a diagnostic, not a rough-sketch
    realism model. No dilation, random displacement, quantization or threshold.
    """
    alpha = (1 - np.asarray(clean[..., 0], np.float64)) / 2
    value = np.fft.irfft2(np.fft.rfft2(alpha) * transfer(alpha.shape, core), s=alpha.shape)
    return np.asarray(1 - 2 * OPACITY * value, np.float32)[..., None]


def inverse(source):
    alpha = (1 - np.asarray(source[..., 0], np.float64)) / (2 * OPACITY)
    value = np.fft.irfft2(np.fft.rfft2(alpha) / transfer(alpha.shape), s=alpha.shape)
    return np.asarray(1 - 2 * value, np.float32)[..., None]


def strength_map(mode, size=SIZE):
    if mode not in MODES:
        raise ValueError(mode)
    t = np.full((size, size, 1), 0.1 if mode == "whole01" else 0.5, np.float32)
    if mode == "partial05":
        y, x = np.mgrid[:size, :size]
        t[(x + 0.3 * y) < 0.60 * size] = 0
    return t


def probes_for(metadata, size=SIZE, margin=24):
    probes, connections, anchors = [], [], []
    for index, path in enumerate(metadata.get("path_geometry", [])):
        points = np.asarray(path["points"]) + margin
        widths = np.asarray(path["widths"])
        normal = normals(points)
        for j in (16, 40, 64, 88, 112):
            if np.all(points[j] >= 16) and np.all(points[j] < size - 16):
                probes.append([*points[j], *normal[j], max(6, widths[j] * 1.5 + 2), index])
        a, b = points[32], points[96]
        if np.all(np.concatenate((a, b)) >= 2) and np.all(np.concatenate((a, b)) < size - 2):
            connections.append([*a, *b, 1])
            anchors.append(points[64])
    if len(anchors) > 1:
        kind = metadata["kinds"][0]
        if kind in ("parallel", "gap", "junction", "crossing"):
            for other in anchors[1:]:
                connections.append([*anchors[0], *other, int(kind in ("junction", "crossing"))])
    return np.asarray(probes, np.float32).reshape(-1, 6), np.asarray(
        connections, np.float32
    ).reshape(-1, 5)


def make_case(split, index):
    if split not in ("train", "probe") or not 0 <= index < COUNT:
        raise ValueError((split, index))
    key = [SEED, int(split == "probe"), index]
    rng = np.random.default_rng(key)
    cfg = GeometryDataConfig(minimum_width=0.6, maximum_width=8, blank_probability=0)
    if index < 72:
        kind = KINDS[index // 8]
        gray, metadata = draw_geometry(rng, SIZE - 48, cfg, kind=kind, include_paths=True)
    elif index < 80:
        kind = "blank" if index < 76 else "black"
        gray = np.full((SIZE - 48, SIZE - 48, 1), int(kind == "blank"), np.float32)
        metadata = dict(kinds=[kind], paths=0)
    elif index < 88:
        kind = "wide"
        wide = cfg.model_copy(update={"minimum_width": 16, "maximum_width": 48})
        gray, metadata = draw_geometry(rng, SIZE - 48, wide, kind="straight", include_paths=True)
    else:
        kind = "mixed"
        gray, metadata = draw_geometry(rng, SIZE - 48, cfg, include_paths=True)
    clean = np.ones((SIZE, SIZE, 1), np.float32)
    clean[24:-24, 24:-24] = np.clip((2 * gray - 1) * 1.1, -1, 1)
    source = corrupt(clean)
    probes, connections = probes_for(metadata)
    record = dict(
        split=split,
        index=index,
        kind=kind,
        key=key,
        clean_sha256=hashlib.sha256(clean.tobytes()).hexdigest(),
        source_sha256=hashlib.sha256(source.tobytes()).hexdigest(),
        inverse_max_error=float(np.max(np.abs(inverse(source) - clean))),
    )
    return dict(clean=clean, source=source, probes=probes, connections=connections), record
