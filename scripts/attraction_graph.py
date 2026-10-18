"""Research representation: nearby anchors project to the same opaque centerline."""

import numpy as np
from centerline_grid import infer_paths, render
from learned_centerline_model import CHANNELS, DIRECTIONS


def radius(level):
    return max(0.75, 4.0 / 2**level)


def target(segments, widths, rectangles, size, level=0, path_ids=None):
    scale = 2**level
    n, reach = size // scale, radius(level)
    s = np.asarray(segments, np.float64).reshape(-1, 2, 2) / scale
    w = np.asarray(widths, np.float64).reshape(-1, 2) / scale
    if len(s) != len(w) or np.any(w < 0):
        raise ValueError("matching nonnegative widths required")
    ids = infer_paths(s) if path_ids is None else np.asarray(path_ids)
    assert len(ids) == len(s)
    best = np.full((n, n), np.inf)
    owner = np.full((n, n), -1, np.int32)
    arc = np.zeros((n, n))
    out = np.zeros((n, n, CHANNELS), np.float32)
    cumulative = {}
    for line, ww, pid in zip(s, w, ids, strict=True):
        low = np.maximum(np.floor(line.min(0) - reach).astype(int), 0)
        high = np.minimum(np.ceil(line.max(0) + reach).astype(int), n)
        v = line[1] - line[0]
        length = float(np.linalg.norm(v))
        start = cumulative.get(int(pid), 0.0)
        cumulative[int(pid)] = start + length
        if np.any(high <= low):
            continue
        yy, xx = np.mgrid[low[1] : high[1], low[0] : high[0]]
        center = np.stack((xx + 0.5, yy + 0.5), -1)
        t = np.clip(((center - line[0]) * v).sum(-1) / max(length**2, 1e-20), 0, 1)
        point = line[0] + t[..., None] * v
        distance = ((point - center) ** 2).sum(-1)
        region = np.s_[low[1] : high[1], low[0] : high[0]]
        update = (distance < best[region]) & (distance <= reach**2)
        best[region][update] = distance[update]
        owner[region][update] = pid
        arc[region][update] = (start + t * length)[update]
        values = out[region]
        values[..., 0][update] = 1
        values[..., 1:3][update] = (point - center)[update]
        values[..., 3][update] = np.log(np.maximum(ww[0] + t * (ww[1] - ww[0]), 1e-8))[update]
    for k, (dx, dy) in enumerate(DIRECTIONS):
        y0, y1 = max(0, -dy), min(n, n - dy)
        x0, x1 = max(0, -dx), min(n, n - dx)
        a = np.s_[y0:y1, x0:x1]
        b = np.s_[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
        out[..., 4 + k][a] = (
            (owner[a] >= 0) & (owner[a] == owner[b]) & (abs(arc[a] - arc[b]) <= 2.0)
        )
    yy, xx = np.mgrid[:n, :n] + 0.5
    for r in np.asarray(rectangles).reshape(-1, 4) / scale:
        out[..., 8] = np.maximum(
            out[..., 8], (xx >= r[0]) & (xx < r[2]) & (yy >= r[1]) & (yy < r[3])
        )
    represented = set(map(int, np.unique(owner[owner >= 0])))
    return out, dict(
        anchors=int(out[..., 0].sum()),
        edges=int(out[..., 4:8].sum()),
        paths=len(set(map(int, ids))),
        represented_paths=len(represented),
        unrepresented_paths=sorted(set(map(int, ids)) - represented),
        radius=reach,
    )


def logits_from_target(truth, level=0):
    h = np.asarray(truth).copy()
    h[..., [0, 4, 5, 6, 7, 8]] = np.where(h[..., [0, 4, 5, 6, 7, 8]] > 0.5, 20.0, -20.0)
    h[..., 1:3] = np.arctanh(np.clip(h[..., 1:3] / radius(level), -1 + 1e-6, 1 - 1e-6))
    return h


def decode(head, level=0):
    h = np.asarray(head)
    size = h.shape[0]
    assert h.shape == (size, size, CHANNELS)
    occupied = h[..., 0] >= 0
    yy, xx = np.mgrid[:size, :size]
    xy = np.stack((xx + 0.5, yy + 0.5), -1) + radius(level) * np.tanh(h[..., 1:3])
    widths = np.exp(np.clip(h[..., 3], -8, np.log(8)))
    degree = np.zeros_like(occupied, dtype=int)
    segments, ww, edges = [], [], []
    for k, (dx, dy) in enumerate(DIRECTIONS):
        for y, x in np.argwhere(occupied & (h[..., 4 + k] >= 0)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size and occupied[ny, nx]:
                segments.append([xy[y, x], xy[ny, nx]])
                ww.append([widths[y, x], widths[ny, nx]])
                degree[y, x] += 1
                degree[ny, nx] += 1
                edges.append(((int(y), int(x)), (int(ny), int(nx))))
    isolated = occupied & (degree == 0)
    for y, x in np.argwhere(isolated):
        segments.append([xy[y, x], xy[y, x]])
        ww.append([widths[y, x], widths[y, x]])
    image = render(
        np.asarray(segments).reshape(-1, 2, 2),
        np.asarray(ww).reshape(-1, 2),
        np.empty((0, 4)),
        size,
    )
    image[h[..., 8] >= 0] = -1
    nodes = [tuple(map(int, p)) for p in np.argwhere(occupied)]
    parent = {p: p for p in nodes}

    def find(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    for a, b in edges:
        parent[find(a)] = find(b)
    return image, dict(
        vertices=len(nodes),
        edges=len(edges),
        isolated=int(isolated.sum()),
        components=len({find(p) for p in nodes}),
        fill_cells=int((h[..., 8] >= 0).sum()),
    )
