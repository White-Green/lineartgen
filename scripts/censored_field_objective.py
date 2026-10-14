"""Fit distance only around a boundary; require signed margins elsewhere."""

import jax
import jax.numpy as jnp
from flax import nnx
from subpixel_field_data import candidate_fields
from subpixel_field_model import DISTANCE_RANGE, SUBPIXELS, field_forward


def robust(x):
    return jnp.sqrt(x * x + 0.1**2) - 0.1


def field_parts(fields, targets, strengths, *, gradient_weight=0.1, sign_weight=0.25, band=1.0):
    assert 0.125 < band < DISTANCE_RANGE
    b, k = targets[0].shape[:2]
    distances = jnp.zeros((b, k))
    gradients = jnp.zeros((b, k))
    signs = jnp.zeros((b, k))
    weight_sum = 0.0
    for i, (pred, target, t) in enumerate(zip(fields, targets, strengths, strict=True)):
        active = jnp.repeat(jnp.repeat(t > 0, SUBPIXELS, 1), SUBPIXELS, 2).astype(pred.dtype)
        target = jnp.clip(target, -DISTANCE_RANGE, DISTANCE_RANGE)
        # Common denominator and ROI for all candidate shifts prevent a width/position bias.
        near = (abs(target) < 4).any(1).astype(pred.dtype)
        weight = (0.01 + 0.99 * near) * active
        denom = weight.sum((1, 2, 3)) + 1e-6
        residual = pred[:, None] - target
        residual = jnp.where(target >= band, jnp.minimum(pred[:, None] - band, 0), residual)
        residual = jnp.where(target <= -band, jnp.maximum(pred[:, None] + band, 0), residual)
        cost = (robust(residual / DISTANCE_RANGE) * weight[:, None]).sum((2, 3, 4)) / denom[:, None]
        grad_cost = jnp.zeros_like(cost)
        for axis in (2, 3):
            # Target has B,K,H,W,C; pred has B,H,W,C.
            pd = jnp.diff(pred, axis=axis - 1) * SUBPIXELS
            td = jnp.diff(target, axis=axis) * SUBPIXELS
            left = [slice(None)] * 4
            right = left.copy()
            left[axis - 1] = slice(None, -1)
            right[axis - 1] = slice(1, None)
            valid = (
                active[tuple(left)] * active[tuple(right)] * near[tuple(left)] * near[tuple(right)]
            )
            tl = [slice(None)] * 5
            tr = tl.copy()
            tl[axis] = slice(None, -1)
            tr[axis] = slice(1, None)
            boundary = (abs(target[tuple(tl)]) < band) & (abs(target[tuple(tr)]) < band)
            valid_candidates = valid[:, None] * boundary
            # Shared normalization across shifts; no candidate can win by having fewer samples.
            gd = valid_candidates.sum((2, 3, 4)).max(1, keepdims=True) + 1e-6
            grad_cost += (robust(pd[:, None] - td) * valid_candidates).sum((2, 3, 4)) / gd
        inside = (target < 0).astype(pred.dtype) * weight[:, None]
        outside = (target > 0).astype(pred.dtype) * weight[:, None]
        # One denominator per class/sample shared across candidate positions.
        ni = jnp.maximum(inside.sum((2, 3, 4)).max(1, keepdims=True), 1.0)
        no = jnp.maximum(outside.sum((2, 3, 4)).max(1, keepdims=True), 1.0)
        margin = jnp.minimum(abs(target), 0.25)
        interior = (jax.nn.relu(margin + pred[:, None]) * inside).sum((2, 3, 4)) / ni
        exterior = (jax.nn.relu(margin - pred[:, None]) * outside).sum((2, 3, 4)) / no
        signs += 0.5**i * (interior + exterior) * 0.5
        distances += 0.5**i * cost
        gradients += 0.5**i * grad_cost / 2
        weight_sum += 0.5**i
    distances /= weight_sum
    gradients /= weight_sum
    signs /= weight_sum
    total = distances + gradient_weight * gradients + sign_weight * signs
    choice = jnp.argmin(total, axis=1, keepdims=True)
    distance = jnp.take_along_axis(distances, choice, axis=1).mean()
    gradient = jnp.take_along_axis(gradients, choice, axis=1).mean()
    sign = jnp.take_along_axis(signs, choice, axis=1).mean()
    return dict(
        distance=distance,
        geometry_gradient=gradient,
        sign_margin=sign,
        loss=distance + gradient_weight * gradient + sign_weight * sign,
    )


def censored_gradient():
    @nnx.jit
    def gradient(model, prepared, base, directions):
        targets = candidate_fields(base, directions, len(prepared.guides))

        def objective(m):
            fields = field_forward(m, prepared)
            parts = field_parts(fields, targets, prepared.noise_levels)
            return parts["loss"], parts

        (_, parts), grads = nnx.value_and_grad(objective, has_aux=True)(model)
        return parts, grads

    return gradient
