"""A deliberately bounded one-vertex-per-cell centerline representation oracle."""

from collections import defaultdict

import numpy as np


def infer_paths(segments):
    ids = []
    current = -1
    previous = None
    for segment in segments:
        if previous is None or not np.allclose(segment[0], previous, atol=1e-6, rtol=0):
            current += 1
        ids.append(current)
        previous = segment[1]
    return np.asarray(ids, np.int32)


def pieces(a, b, cell):
    """Split exactly at grid crossings; corner crossings do not create fake detours."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    delta = b - a
    cuts = [0.0, 1.0]
    for axis in range(2):
        if abs(delta[axis]) > 1e-12:
            lo, hi = sorted((a[axis], b[axis]))
            boundaries = np.arange(np.floor(lo / cell) + 1, np.ceil(hi / cell)) * cell
            cuts.extend(((boundaries - a[axis]) / delta[axis]).tolist())
    cuts = np.unique(np.round(np.clip(cuts, 0, 1), 12))
    result = []
    for left, right in zip(cuts[:-1], cuts[1:], strict=True):
        if right - left <= 1e-11:
            continue
        mid = a + (left + right) * 0.5 * delta
        result.append((tuple(np.floor(mid / cell).astype(int)), left, right))
    if not result:
        result = [(tuple(np.floor(a / cell).astype(int)), 0.0, 1.0)]
    return result


def segment_distance(a, b):
    def cross(u, v):
        return float(u[0] * v[1] - u[1] * v[0])

    av, bv = a[1] - a[0], b[1] - b[0]
    denominator = cross(av, bv)
    if abs(denominator) > 1e-12:
        delta = b[0] - a[0]
        t, u = cross(delta, bv) / denominator, cross(delta, av) / denominator
        if -1e-10 <= t <= 1 + 1e-10 and -1e-10 <= u <= 1 + 1e-10:
            return 0.0

    def point_distance(point, line):
        v = line[1] - line[0]
        t = np.clip(np.dot(point - line[0], v) / max(float(np.dot(v, v)), 1e-20), 0, 1)
        return float(np.linalg.norm(point - (line[0] + t * v)))

    return min(*(point_distance(p, b) for p in a), *(point_distance(p, a) for p in b))


def encode(segments, widths, cell, path_ids=None):
    segments = np.asarray(segments, np.float64).reshape(-1, 2, 2)
    widths = np.asarray(widths, np.float64).reshape(-1, 2)
    if cell <= 0 or len(segments) != len(widths) or np.any(widths < 0):
        raise ValueError("positive cell, matching nonnegative widths required")
    ids = infer_paths(segments) if path_ids is None else np.asarray(path_ids)
    assert len(ids) == len(segments)
    accum, endpoints = defaultdict(list), defaultdict(list)
    walks = defaultdict(list)
    for k, (line, ww, pid) in enumerate(zip(segments, widths, ids, strict=True)):
        parts = pieces(*line, cell)
        length = float(np.linalg.norm(line[1] - line[0]))
        for key, left, right in parts:
            ends = line[0] + np.asarray([left, right])[:, None] * (line[1] - line[0])
            mid = ends.mean(0)
            width = ww[0] + (left + right) * 0.5 * (ww[1] - ww[0])
            accum[key].append((max(length * (right - left), 1e-12), mid, width, int(pid), ends))
            if not walks[int(pid)] or walks[int(pid)][-1] != key:
                walks[int(pid)].append(key)
        if k == 0 or ids[k - 1] != pid:
            endpoints[parts[0][0]].append((line[0], ww[0]))
        if k == len(segments) - 1 or ids[k + 1] != pid:
            endpoints[parts[-1][0]].append((line[1], ww[1]))
    keys = sorted(accum)
    index = {key: i for i, key in enumerate(keys)}
    vertices, radii, collisions = [], [], []
    endpoint_collisions = 0
    for key in keys:
        entries = accum[key]
        weights = np.asarray([e[0] for e in entries])
        point = np.average([e[1] for e in entries], axis=0, weights=weights)
        width = np.average([e[2] for e in entries], weights=weights)
        unique_ends = {(tuple(p), float(w)) for p, w in endpoints[key]}
        if len(unique_ends) == 1:
            p, w = next(iter(unique_ends))
            point, width = np.asarray(p), w
        elif len(unique_ends) > 1:
            endpoint_collisions += 1
        vertices.append(point)
        radii.append(width)
        groups = defaultdict(list)
        for e in entries:
            groups[e[3]].append(e[4])
        paths = sorted(groups)
        for i, pid in enumerate(paths):
            for qid in paths[i + 1 :]:
                distance = min(segment_distance(a, b) for a in groups[pid] for b in groups[qid])
                if distance > 1e-4:
                    collisions.append(dict(cell=list(key), paths=[pid, qid], separation=distance))
    edges = set()
    for walk in walks.values():
        for a, b in zip(walk[:-1], walk[1:], strict=True):
            assert max(abs(a[j] - b[j]) for j in range(2)) <= 1, (a, b)
            if a != b:
                edges.add(tuple(sorted((index[a], index[b]))))
    vertices = np.asarray(vertices).reshape(-1, 2)
    radii = np.asarray(radii)
    edge_array = np.asarray(sorted(edges), np.int32).reshape(-1, 2)
    degrees = np.bincount(edge_array.ravel(), minlength=len(keys))
    parent = list(range(len(keys)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in edge_array:
        parent[find(int(a))] = find(int(b))
    components = len({find(i) for i in range(len(keys))})
    graph_segments = vertices[edge_array]
    graph_widths = radii[edge_array]
    isolated = np.flatnonzero(degrees == 0)
    if len(isolated):
        graph_segments = np.concatenate(
            (graph_segments, np.repeat(vertices[isolated, None], 2, axis=1))
        )
        graph_widths = np.concatenate((graph_widths, np.repeat(radii[isolated, None], 2, axis=1)))
    return dict(
        vertices=vertices,
        widths=radii,
        edges=edge_array,
        cells=np.asarray(keys).reshape(-1, 2),
        segments=graph_segments,
        segment_widths=graph_widths,
        stats=dict(
            vertices=len(keys),
            edges=len(edges),
            components=components,
            isolated_vertices=len(isolated),
            endpoint_collision_cells=endpoint_collisions,
            noncontact_path_collisions=len(collisions),
            collisions=collisions,
        ),
    )


def render(segments, widths, rectangles, size, supersampling=4):
    """CPU tiled version of DQ's capsule-union coverage, with no trainable opacity."""
    s = supersampling
    alpha = np.zeros((size * s, size * s), np.float64)
    for line, ww in zip(segments, widths, strict=True):
        line, ww = np.asarray(line, np.float64), np.asarray(ww, np.float64)
        pad = max(ww) / 2 + 1 / s
        low = np.maximum(np.floor((line.min(0) - pad) * s).astype(int), 0)
        high = np.minimum(np.ceil((line.max(0) + pad) * s).astype(int), size * s)
        if np.any(high <= low):
            continue
        yy, xx = np.meshgrid(
            (np.arange(low[1], high[1]) + 0.5) / s,
            (np.arange(low[0], high[0]) + 0.5) / s,
            indexing="ij",
        )
        grid = np.stack((xx, yy), -1)
        v = line[1] - line[0]
        t = np.clip(((grid - line[0]) * v).sum(-1) / max(float(v @ v), 1e-12), 0, 1)
        distance = np.sqrt(((grid - (line[0] + t[..., None] * v)) ** 2).sum(-1) + 1e-12)
        radius = 0.5 * (ww[0] + t * (ww[1] - ww[0]))
        coverage = np.where(radius > 0, np.clip(0.5 - s * (distance - radius), 0, 1), 0)
        tile = alpha[low[1] : high[1], low[0] : high[0]]
        np.maximum(tile, coverage, out=tile)
    yy, xx = np.meshgrid(
        (np.arange(size * s) + 0.5) / s, (np.arange(size * s) + 0.5) / s, indexing="ij"
    )
    for rect in rectangles:
        distance = np.maximum(
            np.maximum(rect[0] - xx, xx - rect[2]), np.maximum(rect[1] - yy, yy - rect[3])
        )
        np.maximum(alpha, np.clip(0.5 - s * distance, 0, 1), out=alpha)
    return (1 - 2 * alpha.reshape(size, s, size, s).mean((1, 3))).astype(np.float32)[..., None]
