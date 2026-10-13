"""Deterministic structural and ordinary patches for the training-only critic."""

import jax.numpy as jnp
import numpy as np
from path_profile_data import MAX_PATHS, path_samples

MAX_ANCHORS = MAX_PATHS * 3 + MAX_PATHS * (MAX_PATHS - 1) // 2


def structure_anchors(metadata):
    paths = metadata.get("path_geometry", [])
    assert len(paths) <= MAX_PATHS
    sampled = [path_samples(p, np.linspace(0, 1, 65))[0] for p in paths]
    anchors = [p[i] for p in sampled for i in (0, 32, 64)]
    for i, a in enumerate(sampled):
        for b in sampled[i + 1 :]:
            distances = ((a[:, None] - b[None]) ** 2).sum(-1)
            ai, bi = np.unravel_index(distances.argmin(), distances.shape)
            anchors.append((a[ai] + b[bi]) / 2)
    xy = np.zeros((MAX_ANCHORS, 2), np.float32)
    valid = np.zeros(MAX_ANCHORS, bool)
    xy[: len(anchors)] = np.asarray(anchors).reshape(-1, 2)
    valid[: len(anchors)] = True
    return dict(anchors=xy, anchor_valid=valid)


def coordinates(prepared, anchors, valid, seed, step):
    result = []
    for level, (guide, t) in enumerate(zip(prepared.guides, prepared.noise_levels, strict=True)):
        guide, t = np.asarray(guide), np.asarray(t)
        size = min(64, guide.shape[1])
        rows = []
        for item in range(len(guide)):
            rng = np.random.default_rng([seed, step, level, item, 0x434F4F52])
            active = t[item, ..., 0] > 0
            ink = np.argwhere(active & (guide[item, ..., 0] < 0.98))
            paper = np.argwhere(active & (guide[item, ..., 0] > 0.999))
            all_active = np.argwhere(active)
            structure = (anchors[item][valid[item]][:, ::-1] + 0.5) / 2**level - 0.5
            # Metadata guides patch selection only. It never enters the generator.
            for points in (structure,) * 4 + (ink, ink, paper, all_active):
                if not len(points):
                    points = all_active
                center = points[rng.integers(len(points))] if len(points) else np.array([0, 0])
                rows.append(np.clip(np.rint(center) - size // 2, 0, guide.shape[1] - size))
        result.append(jnp.asarray(np.asarray(rows, np.int32).reshape(len(guide), 8, 2)))
    return tuple(result)
