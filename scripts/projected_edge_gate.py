"""Reject connections that violate the target's stage-pixel arc-length bound."""

import numpy as np
from attraction_graph import radius
from learned_centerline_model import DIRECTIONS


def gate(head, level=0, maximum=2.0, tolerance=1e-5, *, base_radius=4.0):
    h = np.asarray(head)
    out = h.copy()
    n = h.shape[0]
    yy, xx = np.mgrid[:n, :n]
    xy = np.stack((xx + 0.5, yy + 0.5), -1) + radius(level, base_radius) * np.tanh(h[..., 1:3])
    rejected = 0
    for k, (dx, dy) in enumerate(DIRECTIONS):
        y0, y1 = max(0, -dy), min(n, n - dy)
        x0, x1 = max(0, -dx), min(n, n - dx)
        a = np.s_[y0:y1, x0:x1]
        b = np.s_[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
        invalid = np.linalg.norm(xy[a] - xy[b], axis=-1) > maximum + tolerance
        used = (h[..., 0][a] >= 0) & (h[..., 0][b] >= 0) & (h[..., 4 + k][a] >= 0)
        rejected += int((used & invalid).sum())
        out[..., 4 + k][a][invalid] = -20
    return out, rejected
