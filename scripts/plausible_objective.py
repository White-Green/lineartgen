"""Tolerant source geometry and an output-only, scale-aware adversarial prior."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from lineartgen.diffusion import downsample, recursive_forward_with_preclip
from lineartgen.training.lineart_adversarial import extract_patches

PRIOR_WEIGHT = 0.05
RAMP = 250
R1_GAMMA = 1.0


def blur(x, sigma):
    radius = max(1, math.ceil(3 * sigma))
    k = np.exp(-(np.arange(-radius, radius + 1, dtype=float) ** 2) / (2 * sigma**2))
    k = jnp.asarray(k / k.sum(), dtype=x.dtype)
    for kernel in (k[:, None, None, None], k[None, :, None, None]):
        x = jax.lax.conv_general_dilated(
            x, kernel, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
        )
    return x


def geometry_parts(prediction, source, strength, level):
    active = (strength > 0).astype(prediction.dtype)
    source_alpha = (1 - source) / (0.8)
    alpha = (1 - prediction) / 2
    # Shared source context outside the editable area, no clean target in this loss.
    value = jnp.where(active > 0, alpha, source_alpha)
    sigma = max(0.5, 4 / (2**level))
    observed, reference = blur(value, sigma), blur(source_alpha, sigma)
    radius = max(1, math.ceil(2 / (2**level)))
    envelope = jax.lax.reduce_window(
        reference,
        -jnp.inf,
        jax.lax.max,
        (1, 2 * radius + 1, 2 * radius + 1, 1),
        (1, 1, 1, 1),
        "SAME",
    )
    tolerance = 0.5 * envelope
    denominator = (reference * active).sum((1, 2, 3)) + 0.001 * active.sum((1, 2, 3)) + 1e-6
    band = (jnp.maximum(abs(observed - reference) - tolerance, 0) * active).sum(
        (1, 2, 3)
    ) / denominator
    mass = (alpha * active).sum((1, 2, 3))
    target_mass = (source_alpha * active).sum((1, 2, 3))
    partial = jnp.any(active == 0, axis=(1, 2, 3))
    low = jnp.where(partial, 0.5, 0.75) * target_mass
    high = jnp.where(partial, 1.5, 1.25) * target_mass
    mass_error = (jax.nn.relu(low - mass) + jax.nn.relu(mass - high)) / denominator
    return dict(geometry=band.mean() + mass_error.mean(), band=band.mean(), mass=mass_error.mean())


class LinePrior(nnx.Module):
    def __init__(self, *, rngs):
        widths = (2, 8, 16, 32, 32)
        self.layers = nnx.List(
            [
                nnx.Conv(a, b, (3, 3), strides=(2, 2), padding="SAME", rngs=rngs)
                for a, b in zip(widths[:-1], widths[1:], strict=True)
            ]
        )
        self.head = nnx.Conv(32, 1, (3, 3), padding="SAME", rngs=rngs)

    def __call__(self, ink, scale):
        x = jnp.concatenate((ink, jnp.full_like(ink, scale)), axis=-1)
        for layer in self.layers:
            x = jax.nn.leaky_relu(layer(x), 0.2)
        return self.head(x).mean((1, 2, 3))


def views(image, strength, coordinates, level):
    """Local native crops plus a full-frame view, with correct physical scale.

    Only wholly active crops contribute. Protected gray sources and mask seams
    are not labeled as bad lineart; the global view is excluded for partial masks.
    """
    size = min(64, image.shape[1])
    local = extract_patches((1 - image) / 2, coordinates, size)
    masks = extract_patches((strength > 0).astype(image.dtype), coordinates, size)
    weights = (masks.min((1, 2, 3)) > 0.999).astype(image.dtype)
    global_image = (1 - image) / 2
    reductions = 0
    while global_image.shape[1] > 64:
        global_image = downsample(global_image)
        reductions += 1
    global_weights = jnp.all(strength > 0, axis=(1, 2, 3)).astype(image.dtype)
    return ((local, weights, level / 5), (global_image, global_weights, (level + reductions) / 5))


def weighted(x, w):
    return (x * w).sum() / jnp.maximum(w.sum(), 1)


def prior_loss(critic, predictions, prepared, coordinates):
    total, denominator = jnp.zeros(()), 0.0
    for level, (p, t, c) in enumerate(
        zip(predictions, prepared.noise_levels, coordinates, strict=True)
    ):
        for patch, w, scale in views(p, t, c, level):
            total += 0.5**level * 0.5 * weighted(jax.nn.softplus(-critic(patch, scale)), w)
        denominator += 0.5**level
    return total / denominator


def geometry_loss(predictions, raw, prepared):
    parts = dict(
        geometry=jnp.zeros(()), band=jnp.zeros(()), mass=jnp.zeros(()), range=jnp.zeros(())
    )
    denominator = 0.0
    for level, (p, z, s, t) in enumerate(
        zip(predictions, raw, prepared.guides, prepared.noise_levels, strict=True)
    ):
        values = geometry_parts(p, s, t, level)
        values["range"] = (jax.nn.relu(abs(z) - 1) * (t > 0)).mean()
        for key, value in values.items():
            parts[key] += 0.5**level * value
        denominator += 0.5**level
    parts = {k: v / denominator for k, v in parts.items()}
    parts["loss"] = parts["geometry"] + 0.05 * parts["range"]
    return parts


def generator_gradient(*, adversarial):
    @nnx.jit
    def gradient(model, prepared, coordinates, critic, weight):
        def objective(m):
            p, z = recursive_forward_with_preclip(
                m,
                prepared.noisy,
                prepared.noise_levels,
                guides=prepared.guides,
                progress=prepared.progress,
            )
            parts = geometry_loss(p, z, prepared)
            parts["prior"] = (
                prior_loss(critic, p, prepared, coordinates) if adversarial else jnp.zeros(())
            )
            parts["loss"] += weight * parts["prior"]
            return parts["loss"], parts

        (_, parts), grads = nnx.value_and_grad(objective, has_aux=True)(model)
        return parts, grads

    return gradient


@nnx.jit
def critic_update(critic, optimizer, fake, prepared, coordinates):
    def objective(m):
        total, r1total, denominator = jnp.zeros(()), jnp.zeros(()), 0.0
        for level, (f, r, t, c) in enumerate(
            zip(fake, prepared.targets, prepared.noise_levels, coordinates, strict=True)
        ):
            f = jax.lax.stop_gradient(f)
            for (fp, w, scale), (rp, _, _) in zip(
                views(f, t, c, level), views(r, t, c, level), strict=True
            ):
                scores = m(rp, scale)
                loss = weighted(jax.nn.softplus(-scores) + jax.nn.softplus(m(fp, scale)), w)
                derivatives = jax.grad(lambda images, scale=scale: m(images, scale).sum())(rp)
                penalty = weighted((derivatives**2).sum((1, 2, 3)), w)
                total += 0.5**level * 0.5 * (loss + 0.5 * R1_GAMMA * penalty)
                r1total += 0.5**level * 0.5 * penalty
            denominator += 0.5**level
        return total / denominator, r1total / denominator

    (loss, r1), grads = nnx.value_and_grad(objective, has_aux=True)(critic)
    optimizer.update(critic, grads)
    return dict(critic_loss=loss, critic_r1=r1, critic_gradient_norm=optax.global_norm(grads))


def create_prior():
    prior = LinePrior(rngs=nnx.Rngs(2026101303))
    optimizer = nnx.Optimizer(
        prior,
        optax.chain(optax.clip_by_global_norm(1), optax.adam(1e-4, b1=0.0, b2=0.99)),
        wrt=nnx.Param,
    )
    return prior, optimizer
