"""Charbonnier candidate costs, with the same normalization and recursive-stage weights."""

import jax.numpy as jnp
from flax import nnx
from plausible_objective import geometry_loss
from stable_assignment_objective import assignment_loss

from lineartgen.diffusion import pyramid, recursive_forward_with_preclip


def robust_costs(predictions, candidates, strengths):
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
        cost = ((jnp.sqrt((alpha[:, None] - target) ** 2 + 0.05**2) - 0.05) * active[:, None]).sum(
            (2, 3, 4)
        )
        costs += 0.5**level * cost / denominator[:, None]
        weight_sum += 0.5**level
    return costs / weight_sum


def robust_gradient(method):
    @nnx.jit
    def gradient(model, prepared, candidates, choices):
        def objective(m):
            predictions, raw = recursive_forward_with_preclip(
                m,
                prepared.noisy,
                prepared.noise_levels,
                guides=prepared.guides,
                progress=prepared.progress,
            )
            parts = geometry_loss(predictions, raw, prepared)
            costs = robust_costs(predictions, candidates, prepared.noise_levels)
            parts["family"] = assignment_loss(costs, choices, method)
            parts["loss"] += parts["family"]
            return parts["loss"], parts

        (_, parts), grads = nnx.value_and_grad(objective, has_aux=True)(model)
        return parts, grads

    return gradient
