"""Deterministic interpolation of aligned, same-architecture inference checkpoints."""

import copy

import jax
import numpy as np
from flax import nnx


def blend_parameters(first, second, alpha):
    """Blend Param leaves on the host, rounding once back to the original dtype.

    This constructs one inference model. It does not combine output images or
    create optimizer/EMA history for a new training run. Endpoints are exact.
    """
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Interpolation alpha must be finite and in [0, 1]")
    left, right = nnx.state(first, nnx.Param), nnx.state(second, nnx.Param)
    if jax.tree.structure(left) != jax.tree.structure(right):
        raise ValueError("Parameter structures differ")

    def blend(a, b):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape or a.dtype != b.dtype or a.dtype.kind != "f":
            raise ValueError("Parameter shapes or floating dtypes differ")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Nonfinite checkpoint parameters")
        if alpha == 0:
            return a.copy()
        if alpha == 1:
            return b.copy()
        return ((1 - alpha) * a.astype(np.float64) + alpha * b.astype(np.float64)).astype(a.dtype)

    state = jax.tree.map(blend, left, right)
    model = nnx.clone(first)
    nnx.update(model, jax.device_put(state))
    return model


def compatible_config(first, second):
    """Require identical UNet forward settings; clip derivative alone may differ."""
    a, b = copy.deepcopy(first.model_dump()), copy.deepcopy(second.model_dump())
    if a["architecture"] != "unet" or b["architecture"] != "unet":
        raise ValueError("Only aligned single UNet checkpoints are supported")
    a.pop("output_clip_gradient")
    b.pop("output_clip_gradient")
    if a != b:
        raise ValueError("Forward model settings differ")
