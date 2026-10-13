"""Balance true endpoint/junction regions without selecting separate local answers."""

import jax.numpy as jnp
from augmented_family_objective import linear_penalty
from flax import nnx
from style_family_objective import style_path_costs, style_pixel_costs

from lineartgen.diffusion import pyramid, recursive_forward_with_preclip


def structural_costs(predictions, candidates, strengths, annotation):
    b, k, h, w, c = candidates.shape
    levels = pyramid(candidates.reshape(b * k, h, w, c), len(predictions))
    normalizers = pyramid(annotation["normalizers"].reshape(b * k, h, w, c), len(predictions))
    regions = pyramid(annotation["region"].astype(jnp.float32), len(predictions))
    costs = jnp.zeros((b, k))
    weight_sum = 0.0
    for level, (prediction, clean, strength, normalization, region) in enumerate(
        zip(predictions, levels, strengths, normalizers, regions, strict=True)
    ):
        clean = clean.reshape(b, k, *prediction.shape[1:])
        active = region * (strength > 0).astype(prediction.dtype)
        alpha, target = (1 - prediction) / 2, (1 - clean) / 2
        normalization = (1 - normalization.reshape(b, k, *prediction.shape[1:])) / 2
        ink = (normalization * active[:, None]).sum((2, 3, 4)).mean(1)
        denominator = ink + 0.001 * active.sum((1, 2, 3)) + 1e-6
        error = (linear_penalty(alpha[:, None], target) * active[:, None]).sum((2, 3, 4))
        costs += 0.5**level * error / denominator[:, None]
        weight_sum += 0.5**level
    return costs / weight_sum


def structural_gradient(coefficient):
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
            normalizers = annotation["normalizers"]
            pixels = style_pixel_costs(raw, candidates, prepared.noise_levels, normalizers)
            profiles = style_path_costs(
                predictions, candidates, prepared.noise_levels, annotation, normalizers
            )
            structure = (
                structural_costs(raw, candidates, prepared.noise_levels, annotation)
                if coefficient
                else jnp.zeros_like(pixels)
            )
            # One candidate for the entire image, all regions, and all stages.
            costs = pixels + 0.25 * profiles + coefficient * structure
            choice = costs.argmin(1)[:, None]
            bound = sum(
                0.5**level * (jnp.maximum(abs(z) - 1, 0) * (t > 0)).mean()
                for level, (z, t) in enumerate(zip(raw, prepared.noise_levels, strict=True))
            ) / sum(0.5**level for level in range(len(raw)))
            parts = dict(
                family=jnp.take_along_axis(pixels, choice, 1).mean(),
                profile=jnp.take_along_axis(profiles, choice, 1).mean(),
                structure=jnp.take_along_axis(structure, choice, 1).mean(),
                range=bound,
            )
            parts["loss"] = costs.min(1).mean() + 0.05 * bound
            return parts["loss"], parts

        (_, parts), grads = nnx.value_and_grad(objective, has_aux=True)(model)
        return parts, grads

    return gradient
