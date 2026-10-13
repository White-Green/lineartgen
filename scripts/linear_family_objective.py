"""Tangent extension of family Charbonnier at physical opacity bounds."""

import jax.numpy as jnp
from flax import nnx
from path_profile_objective import path_costs
from plausible_objective import geometry_loss
from robust_family_objective import robust_costs

from lineartgen.diffusion import pyramid, recursive_forward_with_preclip


def linear_penalty(alpha, target):
    bounded = jnp.clip(alpha, 0, 1)
    residual = bounded - target
    norm = jnp.sqrt(residual**2 + 0.05**2)
    # Continuous value and first derivative at both physical bounds.
    # This is a true scalar loss, with no surrogate clip derivative.
    return norm - 0.05 + (alpha - bounded) * residual / norm


def linear_costs(predictions, candidates, strengths):
    batch, count, height, width, channels = candidates.shape
    levels = pyramid(candidates.reshape(batch * count, height, width, channels), len(predictions))
    costs = jnp.zeros((batch, count))
    weight_sum = 0.0
    for level, (prediction, clean, strength) in enumerate(
        zip(predictions, levels, strengths, strict=True)
    ):
        clean = clean.reshape(batch, count, *prediction.shape[1:])
        active = (strength > 0).astype(prediction.dtype)
        alpha, target = (1 - prediction) / 2, (1 - clean) / 2
        # Denominator is shared by all candidates of an input, so width cannot
        # change the choice merely by changing its normalization.
        ink = (target * active[:, None]).sum((2, 3, 4)).mean(1)
        denominator = ink + 0.001 * active.sum((1, 2, 3)) + 1e-6
        cost = (linear_penalty(alpha[:, None], target) * active[:, None]).sum((2, 3, 4))
        costs += 0.5**level * cost / denominator[:, None]
        weight_sum += 0.5**level
    return costs / weight_sum


def linear_gradient(method):
    assert method in ("clipped", "linear")
    coefficient = 0.25

    @nnx.jit
    def gradient(model, prepared, candidates, annotation):
        def objective(m):
            predictions, raw = recursive_forward_with_preclip(
                m,
                prepared.noisy,
                prepared.noise_levels,
                guides=prepared.guides,
                progress=prepared.progress,
            )
            parts = geometry_loss(predictions, raw, prepared)
            pixels = (
                linear_costs(raw, candidates, prepared.noise_levels)
                if method == "linear"
                else robust_costs(predictions, candidates, prepared.noise_levels)
            )
            profiles = (
                path_costs(predictions, candidates, prepared.noise_levels, annotation)
                if coefficient
                else jnp.zeros_like(pixels)
            )
            costs = pixels + coefficient * profiles
            choice = costs.argmin(1)[:, None]
            parts["profile"] = jnp.take_along_axis(profiles, choice, 1).mean()
            parts["family"] = jnp.take_along_axis(pixels, choice, 1).mean()
            parts["loss"] += costs.min(1).mean()
            return parts["loss"], parts

        (_, parts), grads = nnx.value_and_grad(objective, has_aux=True)(model)
        return parts, grads

    return gradient
