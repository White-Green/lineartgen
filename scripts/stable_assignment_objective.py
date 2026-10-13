"""Test whether a noise/depth-independent clean answer avoids conflicting updates."""

import jax.numpy as jnp
from family_objective import family_costs
from flax import nnx
from plausible_objective import geometry_loss

from lineartgen.diffusion import recursive_forward_with_preclip


def assignment_loss(costs, choices, method):
    assert method in ("dynamic", "stable")
    return (
        costs.min(1)
        if method == "dynamic"
        else jnp.take_along_axis(costs, choices[:, None], axis=1)[:, 0]
    ).mean()


def assignment_gradient(method):
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
            costs = family_costs(predictions, candidates, prepared.noise_levels)
            parts["family"] = assignment_loss(costs, choices, method)
            parts["loss"] += parts["family"]
            return parts["loss"], parts

        (_, parts), grads = nnx.value_and_grad(objective, has_aux=True)(model)
        return parts, grads

    return gradient
