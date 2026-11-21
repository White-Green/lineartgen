"""Distances along observed strokes from ink already painted by the user."""

import numpy as np

from .latent_context import max_pool, repeat2
from .latent_painted_loops import loop_features


def travel(ink, seeds):
    """Eight-neighbor geodesic length and straight-line distance to its source."""
    h, w = ink.shape
    padded = np.pad(ink, 1)
    eligible = padded.ravel().copy()
    fringe = np.flatnonzero(np.pad(seeds & ink, 1)).astype(np.int32)
    distance = np.full(eligible.size, -1, np.int32)
    origin = np.zeros(eligible.size, np.int32)
    stride = w + 2
    neighbors = np.array(
        [-stride - 1, -stride, -stride + 1, -1, 1, stride - 1, stride, stride + 1], np.int32
    )
    origin[fringe] = fringe
    level = 0
    while fringe.size:
        eligible[fringe] = False
        distance[fringe] = level
        targets = fringe[:, None] + neighbors
        valid = eligible[targets]
        expanded, first = np.unique(targets[valid], return_index=True)
        if expanded.size:
            source_rows = np.nonzero(valid)[0][first]
            origin[expanded] = origin[fringe[source_rows]]
        fringe = expanded
        level += 1
    idx = np.arange(eligible.size, dtype=np.int32)
    chord = np.maximum(
        np.abs(idx // stride - origin // stride), np.abs(idx % stride - origin % stride)
    )
    crop = (slice(1, h + 1), slice(1, w + 1))
    return tuple(v.reshape(padded.shape)[crop].astype(np.float32) for v in (distance, chord))


def path_features(rough, coverage):
    h, w = rough.shape
    if not np.any(coverage > 0.08):
        return np.zeros((h, w, 6), np.float32)

    def cells(a):
        a = np.pad(a, ((0, h % 2), (0, w % 2)))
        return a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2).max((1, 3))

    marks = cells(coverage)
    peak = cells(rough)
    local = max_pool(peak, 9)
    ink = peak > np.maximum(0.035, 0.30 * local)
    seeds = marks > np.maximum(0.08, 0.42 * max_pool(marks, 9))
    distance, chord = travel(ink, seeds)
    reached = distance >= 0
    detour = np.maximum(0, distance - chord)
    fields = [reached, detour / np.maximum(3, distance), np.log1p(detour) / 5]
    cap = np.float32(2 * max(peak.shape) + 1)
    # A neighboring branch can be close in the image but far along the rough.
    inverse = np.where(reached, cap - distance, 0)
    for radius in (4, 12):
        advantage = np.maximum(0, distance - (cap - max_pool(inverse, 2 * radius + 1)) - radius)
        fields.append(np.log1p(advantage) / 5)
    fields.append(np.log1p(np.maximum(0, distance)) / 6)
    data = np.stack(fields, -1).astype(np.float32) * reached[..., None]
    return repeat2(data)[:h, :w]


def edit_features(edit, proposal, *, paths=False, loops=False, tails=False):
    local = edit.features(proposal)
    if not paths:
        return local
    rough = edit.rough
    peaks = np.stack([max_pool(rough, k) for k in (3, 7, 15)], -1)
    fields = [local, path_features(rough, edit.coverage), peaks]
    if loops:
        fields.append(loop_features(rough, edit.coverage, tails=tails))
    return np.concatenate(fields, -1)
