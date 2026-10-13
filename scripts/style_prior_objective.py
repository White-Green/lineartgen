"""Fixed clean style family plus an output-only adversarial line prior."""

import jax.numpy as jnp
from flax import nnx
from plausible_objective import prior_loss
from style_family_objective import style_path_costs, style_pixel_costs

from lineartgen.diffusion import recursive_forward_with_preclip


def style_prior_gradient(*, adversarial):
    @nnx.jit
    def gradient(model, prepared, candidates, annotation, critic):
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
            parts["prior"] = (
                prior_loss(critic, predictions, prepared, annotation["coordinates"])
                if adversarial
                else jnp.zeros(())
            )
            parts["loss"] = costs.min(1).mean() + 0.05 * bound
            parts["loss"] += annotation["prior_weight"] * parts["prior"]
            return parts["loss"], parts

        (_, parts), grads = nnx.value_and_grad(objective, has_aux=True)(model)
        return parts, grads

    return gradient
