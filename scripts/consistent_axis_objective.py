"""GX: retain identical candidate costs while restricting complete admissible modes."""

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
    fields,
    targets,
    strengths,
    *,
    gradient_weight=0.1,
    sign_weight=0.25,
    band=1.0,
    sign_power=4,
    candidate_mask=None,
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
    if candidate_mask is not None:
        assert candidate_mask.shape == total.shape
        total = jnp.where(candidate_mask, total, jnp.inf)
    choice = jnp.argmin(total, axis=1, keepdims=True)
    distance = jnp.take_along_axis(distances, choice, axis=1).mean()
    gradient = jnp.take_along_axis(gradients, choice, axis=1).mean()
    sign = jnp.take_along_axis(signs, choice, axis=1).mean()
    return dict(
        selected_alternate_fraction=(choice >= k // 2).astype(jnp.float32).mean(),
        selected_alternate_bits=jnp.sum(
            (choice[:, 0] >= k // 2).astype(jnp.float32) * 2 ** jnp.arange(b)
        ),
        distance=distance,
        geometry_gradient=gradient,
        sign_margin=sign,
        loss=distance + gradient_weight * gradient + sign_weight * sign,
    )


def preferred_original(normal):
    """Choose the more horizontal tangent; near45-degree tie chooses positive slope."""
    nx, ny = normal[..., 0], normal[..., 1]
    tie = jnp.abs(jnp.abs(nx) - jnp.abs(ny)) <= 1e-6
    return (jnp.abs(ny) > jnp.abs(nx) + 1e-6) | (tie & (nx * ny <= 0))


def candidate_mask(normal, strength, policy):
    assert policy in ("winner", "consistent")
    whole = jnp.all(strength > 0, axis=(1, 2, 3))
    original = preferred_original(normal) | ~whole
    if policy == "winner":
        return jnp.repeat(jnp.stack((jnp.ones_like(whole), whole), 1), 9, 1)
    return jnp.repeat(jnp.stack((original, ~original), 1), 9, 1)


def assignment_gradient(policy):
    from context_dropout_objective import dropout_forward
    from flax import nnx
    from subpixel_field_data import candidate_fields

    @nnx.jit
    def gradient(model, prepared, payload, normal):
        base, axis, axis_normal, keep = payload
        depth = len(prepared.guides)
        whole = jnp.all(prepared.noise_levels[0] > 0, axis=(1, 2, 3))
        axis = jnp.where(whole[:, None, None], axis, base)
        axis_normal = jnp.where(whole[:, None], axis_normal, normal)
        aa = candidate_fields(base, normal, depth)
        bb = candidate_fields(axis, axis_normal, depth)
        targets = tuple(jnp.concatenate((a, b), 1) for a, b in zip(aa, bb, strict=True))
        mask = candidate_mask(normal, prepared.noise_levels[0], policy)

        def objective(m):
            parts = field_parts(
                dropout_forward(m, prepared, keep),
                targets,
                prepared.noise_levels,
                candidate_mask=mask,
            )
            return parts["loss"], parts

        (_, parts), grads = nnx.value_and_grad(objective, has_aux=True)(model)
        return parts, grads

    return gradient
