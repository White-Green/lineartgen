"""Evaluate the path-free pixel editor on visible ink, with whole-canvas loop context."""

import numpy as np

from .latent_context import max_pool, mean_pool
from .latent_painted_loops import loop_features


def compact_merge(edit, proposal, session, tile=512):
    h, w = proposal.shape
    loops = loop_features(edit.rough, edit.coverage, tails=True, half_resolution=True)
    result = np.zeros_like(proposal)
    # The widest local feature is a 65px pool displaced by 32px.
    halo = 64
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            end_y, end_x = min(h, y + tile), min(w, x + tile)
            bounds = (
                slice(max(0, y - halo), min(h, end_y + halo)),
                slice(max(0, x - halo), min(w, end_x + halo)),
            )
            y0, x0 = bounds[0].start, bounds[1].start
            inner = (slice(y - y0, end_y - y0), slice(x - x0, end_x - x0))
            p, r, c, a, s = (
                v[bounds] for v in (proposal, edit.rough, edit.coverage, edit.alpha, edit.power)
            )
            peak = max_pool(p, 3)
            iy, ix = np.nonzero((peak[inner] > 1 / 65535) | (r[inner] > 0))
            block = result[y:end_y, x:end_x]
            if not len(iy):
                continue
            rows, cols = iy + inner[0].start, ix + inner[1].start
            # 15 observed/loop input channels and 9 context channels derived from base images.
            feats = np.empty((len(iy), 24), np.float32)
            for i, values in enumerate((p, r, c, a, s)):
                feats[:, i] = values[rows, cols]
            for i, k in enumerate((3, 7, 15), 5):
                feats[:, i] = max_pool(r, k)[rows, cols]
            feats[:, 8:15] = loops[(iy + y) // 2, (ix + x) // 2]
            feats[:, 15] = peak[rows, cols]
            feats[:, 16] = mean_pool(p[..., None], 7)[rows, cols, 0]
            for i, (v, k) in enumerate(((c, 9), (c, 31), (a, 31)), 17):
                feats[:, i] = max_pool(v, k)[rows, cols]
            wide = max_pool(c, 65)
            # Out-of-canvas shifted pool centers are explicitly padded with zero in training.
            for i, (dy, dx) in enumerate(((-32, 0), (32, 0), (0, -32), (0, 32)), 20):
                sy, sx = rows + dy, cols + dx
                valid = (sy >= 0) & (sx >= 0) & (sy < wide.shape[0]) & (sx < wide.shape[1])
                feats[:, i] = 0
                feats[valid, i] = wide[sy[valid], sx[valid]]
            for start in range(0, len(iy), 32768):
                chunk = slice(start, start + 32768)
                (ink,) = session.run(None, {"x": feats[chunk]})
                loc = iy[chunk], ix[chunk]
                block[loc] = ink[:, 0]
    return result
