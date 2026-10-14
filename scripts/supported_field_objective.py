"""Balanced geometry supervision with source-supported exterior penalties."""

import jax
import jax.numpy as jnp
from flax import nnx
from subpixel_field_data import candidate_fields
from subpixel_field_model import DISTANCE_RANGE, SUBPIXELS, field_forward

from lineartgen.diffusion import pyramid
from lineartgen.models.local import local_contrast_cue


def robust(x):
    return jnp.sqrt(x * x + 0.1**2) - 0.1


def field_parts(fields, targets, strengths, supports, *, gradient_weight=0.1, sign_weight=0.25):
    b, k = targets[0].shape[:2]
    distances = jnp.zeros((b, k))
    gradients = jnp.zeros((b, k))
    signs = jnp.zeros((b, k))
    weight_sum = 0.0
    for i, (pred, target, t, support) in enumerate(
        zip(fields, targets, strengths, supports, strict=True)
    ):
        active = jnp.repeat(jnp.repeat(t > 0, SUBPIXELS, 1), SUBPIXELS, 2).astype(pred.dtype)
        target = jnp.clip(target, -DISTANCE_RANGE, DISTANCE_RANGE)
        # Common denominator and ROI for all candidate shifts prevent a width/position bias.
        near = (abs(target) < 4).any(1).astype(pred.dtype)
        weight = (0.01 + 0.99 * near) * active
        denom = weight.sum((1, 2, 3)) + 1e-6
        cost = (robust((pred[:, None] - target) / DISTANCE_RANGE) * weight[:, None]).sum(
            (2, 3, 4)
        ) / denom[:, None]
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
            gd = valid.sum((1, 2, 3)) + 1e-6
            grad_cost += (robust(pd[:, None] - td) * valid[:, None]).sum((2, 3, 4)) / gd[:, None]
        inside = (target < 0).astype(pred.dtype) * weight[:, None]
        # Extend only the exterior sign ROI. Distance, gradient and interior terms
        # retain DT weighting. Support is fixed from the input, never prediction.
        support = jnp.repeat(jnp.repeat(support, SUBPIXELS, 1), SUBPIXELS, 2)
        exterior_weight = (0.01 + 0.99 * jnp.maximum(near, support)) * active
        outside = (target > 0).astype(pred.dtype) * exterior_weight[:, None]
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


def source_supports(prepared):
    normalized = local_contrast_cue(prepared.guides[0])
    support = jnp.clip((1 - normalized) / 0.8, 0, 1)
    return pyramid(support, len(prepared.guides))


def supported_gradient():
    @nnx.jit
    def gradient(model, prepared, base, directions):
        targets = candidate_fields(base, directions, len(prepared.guides))
        supports = source_supports(prepared)

        def objective(m):
            fields = field_forward(m, prepared)
            parts = field_parts(fields, targets, prepared.noise_levels, supports)
            return parts["loss"], parts

        (_, parts), grads = nnx.value_and_grad(objective, has_aux=True)(model)
        return parts, grads

    return gradient
