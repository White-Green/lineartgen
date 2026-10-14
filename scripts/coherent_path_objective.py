"""Local clean-profile choices coupled along paths, with shared choices across scales."""

import jax
import jax.numpy as jnp
from augmented_family_objective import linear_penalty
from flax import nnx
from path_profile_objective import sample_batch

from lineartgen.diffusion import pyramid, recursive_forward_with_preclip

SAMPLES = 65
PATHS = 4
OFFSETS = jnp.linspace(-2, 2, 9)
TRANSITION = 0.25


def hard_min(values, axis):
    """Select one minimizing path, including ties, instead of averaging their gradients."""
    index = jnp.argmin(values, axis=axis, keepdims=True)
    return jnp.take_along_axis(values, index, axis=axis).squeeze(axis)


def envelope_cost(raw, candidates, strengths):
    b, k, h, w, c = candidates.shape
    targets = pyramid(candidates.reshape(b * k, h, w, c), len(raw))
    total, weights = jnp.zeros(b), 0.0
    for level, (z, clean, t) in enumerate(zip(raw, targets, strengths, strict=True)):
        target = ((1 - clean) / 2).reshape(b, k, *z.shape[1:])
        alpha = (1 - z) / 2
        low, high = target.min(1), target.max(1)
        nearest = jnp.clip(alpha, low, high)
        active = (t > 0).astype(z.dtype)
        denominator = (target * active[:, None]).sum((2, 3, 4)).mean(1)
        denominator += 0.001 * active.sum((1, 2, 3)) + 1e-6
        value = (linear_penalty(alpha, nearest) * active).sum((1, 2, 3)) / denominator
        total += 0.5**level * value
        weights += 0.5**level
    return total / weights


def path_dynamic_cost(unary, valid, spacing, coefficient=TRANSITION):
    """B,P,S,K costs; masked gaps reset transitions, retaining completed segment costs."""
    assert unary.shape[-1] == 9
    mask = valid.astype(unary.dtype)
    unary = unary * mask[..., None]
    delta2 = (OFFSETS[:, None] - OFFSETS[None, :]) ** 2
    first = unary[:, :, 0]

    def step(previous, inputs):
        local, active, preceding, distance = inputs
        penalty = coefficient * delta2 / jnp.maximum(distance[..., None, None], 1.0)
        penalty *= (active * preceding)[..., None, None]
        return local + hard_min(previous[..., :, None] + penalty, axis=-2), None

    last, _ = jax.lax.scan(
        step,
        first,
        (
            jnp.moveaxis(unary[:, :, 1:], 2, 0),
            jnp.moveaxis(mask[:, :, 1:], 2, 0),
            jnp.moveaxis(mask[:, :, :-1], 2, 0),
            jnp.moveaxis(spacing, 2, 0),
        ),
    )
    counts = mask.sum(-1)
    by_path = hard_min(last, -1) / jnp.maximum(counts, 1)
    present = (counts > 0).astype(unary.dtype)
    return (by_path * present).sum(-1) / jnp.maximum(present.sum(-1), 1)


def global_path_cost(unary, valid):
    counts = valid.sum(-1)
    by_path = (unary * valid[..., None]).sum(-2) / jnp.maximum(counts[..., None], 1)
    present = (counts > 0).astype(unary.dtype)
    costs = (by_path * present[..., None]).sum(1) / jnp.maximum(present.sum(1)[:, None], 1)
    return hard_min(costs, -1)


def profile_unary(raw, candidates, strengths, ann):
    b, k, h, w, c = candidates.shape
    targets = pyramid(candidates.reshape(b * k, h, w, c), len(raw))
    probes, valid = ann["dense_probes"], ann["dense_valid"]
    offsets = jnp.linspace(-1, 1, 129)[None, None, :] * probes[:, :, 4, None]
    xy = jnp.moveaxis(
        probes[:, :, :2, None] + probes[:, :, 2:4, None] * offsets[:, :, None], -2, -1
    )
    total = jnp.zeros((b, PATHS, SAMPLES, k))
    visible_any = jnp.zeros((b, PATHS, SAMPLES), dtype=bool)
    weights = 0.0
    for level, (z, clean, t) in enumerate(zip(raw, targets, strengths, strict=True)):
        scale = 2**level
        q = (xy + 0.5) / scale - 0.5
        active = (t > 0).astype(z.dtype)
        observed = sample_batch(((1 - z) / 2) * active, q)
        reference = sample_batch(
            ((1 - clean) / 2) * jnp.repeat(active, k, 0), jnp.repeat(q, k, 0)
        ).reshape(b, k, *observed.shape[1:])
        visible = valid * jnp.all(sample_batch(active, q) > 0.999, -1)
        dx = 2 * probes[:, :, 4] / (128 * scale)
        mass = jnp.trapezoid(reference, axis=-1) * dx[:, None]
        denominator = jnp.maximum(mass.mean(1), 0.25 / scale)
        error = linear_penalty(observed[:, None], reference)
        cost = jnp.trapezoid(error, axis=-1) * dx[:, None] / denominator[:, None]
        cost *= visible[:, None]
        total += 0.5**level * jnp.moveaxis(cost.reshape(b, k, PATHS, SAMPLES), 1, -1)
        visible_any |= visible.reshape(b, PATHS, SAMPLES) > 0
        weights += 0.5**level
    return total / weights, visible_any


def structured_parts(raw, candidates, strengths, ann, method):
    assert method in ("global", "path")
    envelope = envelope_cost(raw, candidates, strengths)
    unary, valid = profile_unary(raw, candidates, strengths, ann)
    profiles = (
        global_path_cost(unary, valid)
        if method == "global"
        else path_dynamic_cost(unary, valid, ann["spacing"])
    )
    bound = sum(
        0.5**i * (jnp.maximum(abs(z) - 1, 0) * (t > 0)).mean()
        for i, (z, t) in enumerate(zip(raw, strengths, strict=True))
    )
    bound /= sum(0.5**i for i in range(len(raw)))
    return dict(
        envelope=envelope.mean(),
        profile=profiles.mean(),
        range=bound,
        loss=(envelope + profiles).mean() + 0.05 * bound,
    )


def structured_gradient(method):
    @nnx.jit
    def gradient(model, prepared, candidates, annotation):
        def objective(m):
            _, raw = recursive_forward_with_preclip(
                m,
                prepared.noisy,
                prepared.noise_levels,
                guides=prepared.guides,
                progress=prepared.progress,
            )
            parts = structured_parts(raw, candidates, prepared.noise_levels, annotation, method)
            return parts["loss"], parts

        (_, parts), grads = nnx.value_and_grad(objective, has_aux=True)(model)
        return parts, grads

    return gradient
