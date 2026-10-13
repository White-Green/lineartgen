"""Balance clean-family errors by path and arc position, retaining one coherent answer."""

import jax
import jax.numpy as jnp
from flax import nnx
from jax.scipy.ndimage import map_coordinates
from path_profile_data import FRACTIONS, MAX_PATHS
from plausible_objective import geometry_loss
from robust_family_objective import robust_costs

from lineartgen.diffusion import pyramid, recursive_forward_with_preclip


def sample_batch(images, xy):
    return jax.vmap(
        lambda image, q: map_coordinates(
            image[..., 0], [q[..., 1], q[..., 0]], order=1, mode="constant", cval=0
        )
    )(images, xy)


def path_costs(predictions, candidates, strengths, annotation):
    b, k, h, w, c = candidates.shape
    levels = pyramid(candidates.reshape(b * k, h, w, c), len(predictions))
    probes, valid = annotation["probes"], annotation["valid"]
    offsets = jnp.linspace(-1, 1, 129)[None, None, :] * probes[:, :, 4, None]
    xy = probes[:, :, :2, None] + probes[:, :, 2:4, None] * offsets[:, :, None, :]
    xy = jnp.moveaxis(xy, -2, -1)
    total = jnp.zeros((b, k))
    weight_sum = 0.0
    for level, (prediction, clean, strength) in enumerate(
        zip(predictions, levels, strengths, strict=True)
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
        mass = jnp.trapezoid(reference, axis=-1) * dx[:, None, :]
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


def profile_gradient(coefficient):
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
            pixels = robust_costs(predictions, candidates, prepared.noise_levels)
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
