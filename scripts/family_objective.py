"""Finite-family oracle diagnostic: choose one full-image answer, or average costs."""

import jax.numpy as jnp
from flax import nnx
from plausible_objective import geometry_loss

from lineartgen.diffusion import pyramid, recursive_forward_with_preclip


def family_costs(predictions, candidates, strengths):
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
        cost = (((alpha[:, None] - target) ** 2) * active[:, None]).sum((2, 3, 4))
        costs += 0.5**level * cost / denominator[:, None]
        weight_sum += 0.5**level
    return costs / weight_sum


def family_gradient(method):
    assert method in ("mean", "minimum")

    @nnx.jit
    def gradient(model, prepared, candidates):
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
            # Choose after accumulating the whole image AND all stages.
            # Per-pixel or independent per-stage choices would allow hybrids.
            parts["family"] = (costs.min(1) if method == "minimum" else costs.mean(1)).mean()
            parts["loss"] += parts["family"]
            return parts["loss"], parts

        (_, parts), grads = nnx.value_and_grad(objective, has_aux=True)(model)
        return parts, grads

    return gradient
