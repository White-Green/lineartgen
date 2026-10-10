"""Continuous ink-coverage agreement, complementing threshold and topology metrics."""

import numpy as np


def coverage_metrics(prediction, target, strength):
    p = np.clip((1 - prediction[..., 0]) / 2, 0, 1).astype(np.float64)
    y = np.clip((1 - target[..., 0]) / 2, 0, 1).astype(np.float64)
    active = strength[..., 0] > 0
    p, y = p[active], y[active]
    total = float(p.sum() + y.sum())
    error = float(np.abs(p - y).sum())
    return dict(
        coverage_overlap_f1=2 * float(np.minimum(p, y).sum()) / total if total else 1.0,
        active_alpha_l1=float(np.abs(p - y).mean()) if p.size else 0.0,
        active_alpha_l1_sum=error,
        total_predicted_and_target_coverage=total,
        active_pixels=int(p.size),
    )
