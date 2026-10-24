"""Pointwise width selection; geometry and recursive feedback remain unchanged."""

import numpy as np


def gate_features(full, painted, detail, noise, level=0):
    ink = np.clip((1 - detail) / 0.8, 0, 1)
    padded = np.pad(ink, ((0, 0), (1, 1), (1, 1), (0, 0)))
    h, w = ink.shape[1:3]
    nearby = np.maximum.reduce(
        [padded[:, y : y + h, x : x + w] for y in range(3) for x in range(3)]
    )
    return np.concatenate(
        (
            np.clip(full[..., 3:4], -8, 8) / 4,
            np.clip(painted[..., 3:4], -8, 8) / 4,
            nearby,
            noise,
            np.full_like(noise, level / 5),
        ),
        axis=-1,
    )


def fuse_width(session, full, painted, detail, noise, level=0):
    features = gate_features(full, painted, detail, noise, level)
    weight = session.run(["weight"], {"features": features})[0]
    if weight.shape != full.shape[:-1] + (1,) or not np.isfinite(weight).all():
        raise RuntimeError("Width selection produced invalid output")
    if np.any((weight < 0) | (weight > 1)):
        raise RuntimeError("Width selection must be a convex combination")
    weight = np.where(features[..., 2:3] > 0, weight, 0)
    result = full.copy()
    result[..., 3:4] += weight * (painted[..., 3:4] - full[..., 3:4])
    return result
