"""Emphasize sparse wrong-sign defects with a regional fourth-power mean."""

import jax
import jax.numpy as jnp
from subpixel_field_model import DISTANCE_RANGE, SUBPIXELS


def robust(x):
    return jnp.sqrt(x * x + 0.1**2) - 0.1


def sign_mean(violation, mask, denominator, *, power):
    assert power in (2, 4)
    if power == 2:
        return (
            jnp.sqrt((violation * violation * mask).sum((2, 3, 4)) / denominator + 0.01**2) - 0.01
        )
    squared = violation * violation
    return ((squared * squared * mask).sum((2, 3, 4)) / denominator + 0.01**4) ** 0.25 - 0.01


def field_parts(
    fields, targets, strengths, *, gradient_weight=0.1, sign_weight=0.25, band=1.0, sign_power=4
):
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
        interior = jnp.zeros_like(cost)
        exterior = jnp.zeros_like(cost)
        margin = jnp.minimum(abs(target), 0.25)
        vi = jax.nn.relu(margin + pred[:, None])
        vo = jax.nn.relu(margin - pred[:, None])
        for region in (near * active, (1 - near) * active):
            inside = (target < 0).astype(pred.dtype) * region[:, None]
            outside = (target > 0).astype(pred.dtype) * region[:, None]
            ni = jnp.maximum(inside.sum((2, 3, 4)).max(1, keepdims=True), 1.0)
            no = jnp.maximum(outside.sum((2, 3, 4)).max(1, keepdims=True), 1.0)
            # Sum independent requirements. Empty regions contribute exactly zero.
            interior += sign_mean(vi, inside, ni, power=sign_power)
            exterior += sign_mean(vo, outside, no, power=sign_power)
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
