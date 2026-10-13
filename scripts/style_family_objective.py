"""Compare line styles with common original-ink normalization and no source-mass penalty."""

import jax.numpy as jnp
from augmented_family_objective import linear_penalty
from flax import nnx
from path_profile_data import FRACTIONS, MAX_PATHS
from path_profile_objective import sample_batch

from lineartgen.diffusion import pyramid, recursive_forward_with_preclip


def style_pixel_costs(predictions, candidates, strengths, normalizers):
    batch, count, height, width, channels = candidates.shape
    levels = pyramid(candidates.reshape(batch * count, height, width, channels), len(predictions))
    normalizer_levels = pyramid(
        normalizers.reshape(batch * count, height, width, channels), len(predictions)
    )
    costs = jnp.zeros((batch, count))
    weight_sum = 0.0
    for level, (prediction, clean, strength, normalization) in enumerate(
        zip(predictions, levels, strengths, normalizer_levels, strict=True)
    ):
        clean = clean.reshape(batch, count, *prediction.shape[1:])
        active = (strength > 0).astype(prediction.dtype)
        alpha, target = (1 - prediction) / 2, (1 - clean) / 2
        # Denominator is shared by all candidates of an input, so width cannot
        # change the choice merely by changing its normalization.
        normalization = (1 - normalization.reshape(batch, count, *prediction.shape[1:])) / 2
        ink = (normalization * active[:, None]).sum((2, 3, 4)).mean(1)
        denominator = ink + 0.001 * active.sum((1, 2, 3)) + 1e-6
        cost = (linear_penalty(alpha[:, None], target) * active[:, None]).sum((2, 3, 4))
        costs += 0.5**level * cost / denominator[:, None]
        weight_sum += 0.5**level
    return costs / weight_sum


def style_path_costs(predictions, candidates, strengths, annotation, normalizers):
    b, k, h, w, c = candidates.shape
    levels = pyramid(candidates.reshape(b * k, h, w, c), len(predictions))
    normalizer_levels = pyramid(normalizers.reshape(b * k, h, w, c), len(predictions))
    probes, valid = annotation["probes"], annotation["valid"]
    offsets = jnp.linspace(-1, 1, 129)[None, None, :] * probes[:, :, 4, None]
    xy = probes[:, :, :2, None] + probes[:, :, 2:4, None] * offsets[:, :, None, :]
    xy = jnp.moveaxis(xy, -2, -1)
    total = jnp.zeros((b, k))
    weight_sum = 0.0
    for level, (prediction, clean, strength, normalization) in enumerate(
        zip(predictions, levels, strengths, normalizer_levels, strict=True)
    ):
        scale = 2**level
        q = (xy + 0.5) / scale - 0.5
        active = (strength > 0).astype(prediction.dtype)
        obs = sample_batch(((1 - prediction) / 2) * active, q)
        reference = sample_batch(
            ((1 - clean) / 2) * jnp.repeat(active, k, axis=0), jnp.repeat(q, k, axis=0)
        ).reshape(b, k, *obs.shape[1:])
        visible = valid * jnp.all(sample_batch(active, q) > 0.999, axis=-1)
        dx = 2 * probes[:, :, 4] / (128 * scale)
        normalization = sample_batch(
            ((1 - normalization) / 2) * jnp.repeat(active, k, axis=0), jnp.repeat(q, k, axis=0)
        ).reshape(b, k, *obs.shape[1:])
        mass = jnp.trapezoid(normalization, axis=-1) * dx[:, None, :]
        denominator = jnp.maximum(mass.mean(1), 0.25 / scale)
        error = jnp.sqrt((obs[:, None] - reference) ** 2 + 0.05**2) - 0.05
        error = jnp.trapezoid(error, axis=-1) * dx[:, None, :] / denominator[:, None, :]
        visible = visible.reshape(b, MAX_PATHS, len(FRACTIONS))
        error = error.reshape(b, k, MAX_PATHS, len(FRACTIONS))
        counts = visible.sum(-1)
        by_path = (error * visible[:, None]).sum(-1) / jnp.maximum(counts[:, None], 1)
        present = (counts > 0).astype(prediction.dtype)
        by_image = (by_path * present[:, None]).sum(-1) / jnp.maximum(
            present.sum(-1, keepdims=True), 1
        )
        total += 0.5**level * by_image
        weight_sum += 0.5**level
    return total / weight_sum


def style_gradient():
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
            costs = pixels + 0.25 * profiles
            choice = costs.argmin(1)[:, None]
            bound = sum(
                0.5**level * (jnp.maximum(abs(z) - 1, 0) * (t > 0)).mean()
                for level, (z, t) in enumerate(zip(raw, prepared.noise_levels, strict=True))
            ) / sum(0.5**level for level in range(len(raw)))
            parts = dict(
                family=jnp.take_along_axis(pixels, choice, 1).mean(),
                profile=jnp.take_along_axis(profiles, choice, 1).mean(),
                range=bound,
            )
            parts["loss"] = costs.min(1).mean() + 0.05 * bound
            return parts["loss"], parts

        (_, parts), grads = nnx.value_and_grad(objective, has_aux=True)(model)
        return parts, grads

    return gradient
