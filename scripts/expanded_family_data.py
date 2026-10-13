"""Increase the allowed-family corpus while retaining the original 64 cases."""

import hashlib

import numpy as np
from cue_preserving_data import probes_for
from plausible_data import CHOICES, ENDPOINTS, OPACITY, SIZE, profile_samples, raster
from scipy import ndimage

from lineartgen.config import GeometryDataConfig
from lineartgen.data.geometry import KINDS, draw_geometry, normals

SEED = 2026101306


def make_case(split, slot, variant):
    assert split in ("expanded", "new_probe") and 0 <= slot < 64
    key = [SEED, int(split == "new_probe"), slot, variant]
    rng = np.random.default_rng(key)
    kind = (
        KINDS[slot // 6]
        if slot < 54
        else "blank"
        if slot < 56
        else "black"
        if slot < 58
        else "wide"
    )
    config = GeometryDataConfig(minimum_width=0.8, maximum_width=5, blank_probability=0)
    if kind in ("blank", "black"):
        image = np.ones((SIZE, SIZE, 1), np.float32)
        if kind == "black":
            image[32:-32, 32:-32] = -1
        candidates = np.stack([image] * len(CHOICES))
        metadata = dict(kinds=[kind], paths=0)
        direction = np.array([1.0, 0.0])
    else:
        if kind == "wide":
            config = config.model_copy(update={"minimum_width": 16, "maximum_width": 32})
        _, metadata = draw_geometry(
            rng, SIZE - 64, config, kind="straight" if kind == "wide" else kind, include_paths=True
        )
        direction = normals(np.asarray(metadata["path_geometry"][0]["points"]))[64]
        candidates = np.stack(
            [raster(metadata, direction, offset, width) for offset, width in CHOICES]
        )
    mixture = candidates[list(ENDPOINTS)].mean(0)
    source = 1 - 2 * OPACITY * ndimage.gaussian_filter(
        (1 - mixture) / 2, (1.2, 1.2, 0), mode="constant", cval=0
    )
    probes, connections = probes_for(metadata, margin=32)
    if len(probes):
        probes[:, 4] += 4
    data = dict(
        variants=candidates,
        source=source.astype(np.float32),
        mixture=mixture,
        probes=probes,
        connections=connections,
        direction=direction,
        profiles=np.stack([profile_samples(x, probes) for x in candidates]),
    )
    record = dict(
        split=split,
        slot=slot,
        variant=variant,
        key=key,
        kind=kind,
        metadata=metadata,
        direction=direction.tolist(),
        variants_sha256=hashlib.sha256(candidates.tobytes()).hexdigest(),
        source_sha256=hashlib.sha256(data["source"].tobytes()).hexdigest(),
    )
    return data, record
