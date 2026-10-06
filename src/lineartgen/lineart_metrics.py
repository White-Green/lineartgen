"""Continuous-opacity diagnostics; gray halos cannot hide behind a 50% ink threshold."""

import numpy as np
from scipy import ndimage


def halo_metrics(prediction, target, strength, radius=6):
    p = np.clip((1 - prediction[..., 0]) / 2, 0, 1)
    y = np.clip((1 - target[..., 0]) / 2, 0, 1)
    active = strength[..., 0] > 0
    support = y > 0.02
    distance = (
        ndimage.distance_transform_edt(~support) if support.any() else np.full(y.shape, np.inf)
    )
    near = active & ~support & (distance <= radius)
    far = active & ~support & (distance > radius)
    extra = np.maximum(p - y, 0)
    core = active & support

    def mean(values, mask):
        return float(values[mask].mean()) if mask.any() else None

    result = dict(
        halo_pixels=int(near.sum()),
        halo_opacity_excess=mean(extra, near),
        halo_visible_fraction=mean(p >= 0.1, near),
        halo_gray_fraction=mean((p >= 0.1) & (p < 0.5), near),
        far_opacity_excess=mean(extra, far),
        ink_deficit_fraction=float(np.maximum(y - p, 0)[core].sum() / y[core].sum())
        if y[core].sum() > 1e-6
        else None,
        protected_max_error=float(np.abs(prediction[..., 0] - target[..., 0])[~active].max())
        if (~active).any()
        else None,
    )
    for threshold in (0.1, 0.25, 0.5, 0.75):
        predicted, truth = (p > threshold) & active, (y > threshold) & active
        count = int(predicted.sum()) + int(truth.sum())
        result[f"ink_f1_alpha{round(threshold * 100):02d}"] = (
            2 * int((predicted & truth).sum()) / count if count else 1.0
        )
    return result
