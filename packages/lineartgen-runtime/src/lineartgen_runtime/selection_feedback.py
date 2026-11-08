"""CNN selection projected onto the original graph components."""

import numpy as np


class TiledSelectionSession:
    def __init__(self, session, core=512, halo=192):
        if core < 8 or core % 8 or halo < 192 or halo % 8:
            raise ValueError("Aligned cores and at least 192 pixels of halo are required")
        self.session, self.core, self.halo = (session, core, halo)

    def run(self, names, inputs):
        if names != ["graph", "state", "base_graph"] or set(inputs) != {"features", "latent"}:
            raise ValueError("Invalid selection stage inputs/outputs")
        f, s = (inputs["features"], inputs["latent"])
        _, h, w, _ = f.shape
        if f.shape != (1, h, w, 10) or s.shape != (1, h, w, 16):
            raise ValueError("Invalid selection stage shapes")
        if max(h, w) <= self.core or h % 8 or w % 8:
            return self.session.run(names, inputs)
        outputs = [np.empty((1, h, w, c), np.float32) for c in (9, 16, 9)]
        for y in range(0, h, self.core):
            y1 = min(h, y + self.core)
            top, bottom = (max(0, y - self.halo), min(h, y1 + self.halo))
            for x in range(0, w, self.core):
                x1 = min(w, x + self.core)
                left, right = (max(0, x - self.halo), min(w, x1 + self.halo))
                crop = {
                    k: np.ascontiguousarray(v[:, top:bottom, left:right]) for k, v in inputs.items()
                }
                result = self.session.run(names, crop)
                for out, tile, c in zip(outputs, result, (9, 16, 9), strict=True):
                    if (
                        tile.shape != (1, bottom - top, right - left, c)
                        or not np.isfinite(tile).all()
                    ):
                        raise RuntimeError("Invalid selection tile output")
                    out[:, y:y1, x:x1] = tile[:, y - top : y1 - top, x - left : x1 - left]
        return outputs


DIRECTIONS = ((1, 0), (0, 1), (1, 1), (-1, 1))


def components(base):
    if base.ndim != 3 or base.shape[-1] != 9 or (not np.isfinite(base).all()):
        raise ValueError("Expected a finite HWC nine-channel graph")
    h, w, _ = base.shape
    occupied = base[..., 0] >= 0
    active = np.flatnonzero(occupied)
    index = np.full((h, w), -1, np.int32)
    index.flat[active] = np.arange(len(active), dtype=np.int32)
    parent = np.arange(len(active), dtype=np.int32)
    size = np.ones(len(active), np.int32)
    yy, xx = np.mgrid[:h, :w]
    xy = np.stack((xx + 0.5, yy + 0.5), -1) + np.tanh(base[..., 1:3])

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for k, (dx, dy) in enumerate(DIRECTIONS):
        y0, y1 = (max(0, -dy), min(h, h - dy))
        x0, x1 = (max(0, -dx), min(w, w - dx))
        a = np.s_[y0:y1, x0:x1]
        b = np.s_[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
        valid = occupied[a] & occupied[b] & (base[..., 4 + k][a] >= 0)
        valid &= np.linalg.norm(xy[a] - xy[b], axis=-1) <= 2.0 + 1e-05
        for i, j in zip(index[a][valid], index[b][valid], strict=True):
            i, j = (find(i), find(j))
            if i != j:
                if size[i] < size[j]:
                    i, j = (j, i)
                parent[j] = i
                size[i] += size[j]
    roots = np.fromiter((find(i) for i in range(len(active))), dtype=np.int32, count=len(active))
    _, labels = np.unique(roots, return_inverse=True)
    result = np.full((h, w), -1, np.int32)
    result.flat[active] = labels
    return result


def project(head, base, threshold=0.0):
    if head.shape != base.shape or not np.isfinite(head).all() or (not np.isfinite(threshold)):
        raise ValueError("Expected matching finite CNN and base graphs")
    ids = components(base)
    valid = ids >= 0
    result = base.copy()
    if not valid.any():
        return result
    labels = ids[valid]
    count = np.bincount(labels)
    score = np.bincount(labels, weights=head[..., 0][valid]) / count
    removed = score[labels] < threshold
    values = result[..., 0][valid]
    values[removed] = -20.0
    result[..., 0][valid] = values
    return result


def forward(session, levels):
    from lineartgen_runtime.stage_feedback import features, resize_linear

    if not 1 <= len(levels) <= 6:
        raise ValueError("Expected one to six levels")
    heads, previous, state = ([None] * len(levels), None, None)
    for level in reversed(range(len(levels))):
        value = features(levels[level][..., :5], previous, level)
        _, h, w, _ = value.shape
        latent = (
            np.zeros((1, h, w, 16), np.float32) if state is None else resize_linear(state, h, w)
        )
        head, state, previous = session.run(
            ["graph", "state", "base_graph"], {"features": value, "latent": latent}
        )
        if (
            head.shape != (1, h, w, 9)
            or state.shape != (1, h, w, 16)
            or previous.shape != head.shape
        ):
            raise RuntimeError("Invalid component stage shape")
        if any(not np.isfinite(v).all() for v in (head, state, previous)):
            raise RuntimeError("Nonfinite component stage output")
        heads[level] = head
    heads[0] = project(heads[0][0], previous[0])[None]
    return tuple(heads)
