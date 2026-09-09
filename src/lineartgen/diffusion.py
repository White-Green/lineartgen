"""Forward noise, image pyramids and recursive conditioning; independent of the trainer."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from lineartgen.config import DiffusionConfig
from lineartgen.noise_matching import match_noise


class TrainingInputs(NamedTuple):
    targets: tuple
    noise_levels: tuple
    noisy: tuple


def downsample(image):
    batch, height, width, channels = image.shape
    return image.reshape(batch, height // 2, 2, width // 2, 2, channels).mean((2, 4))


def pyramid(image, levels):
    images = [image]
    for _ in range(levels - 1):
        images.append(downsample(images[-1]))
    return tuple(images)


def recursive_forward(model, images, noise_levels):
    smallest = images[-1]
    insert = jnp.zeros((smallest.shape[0], smallest.shape[1] // 2, smallest.shape[2] // 2, 1))
    outputs = []
    for image, noise in zip(reversed(images), reversed(noise_levels), strict=True):
        insert = model(image, noise, insert)
        outputs.append(insert)
    return tuple(reversed(outputs))


def noise_map(clean, key, config: DiffusionConfig):
    """Full, band, rectangle and region-with-protected-lines patterns, entirely on device."""
    batch, height, width, _ = clean.shape
    random = jax.random.uniform(key, (batch, 10, 1, 1, 1))
    base = config.min_noise + random[:, 0] * (1 - config.min_noise)
    weights = jnp.asarray(config.pattern_weights)
    boundaries = jnp.cumsum(weights) / weights.sum()
    pattern = (random[:, 1, None] > boundaries[None, :, None, None, None]).sum(axis=1)
    # Keep the batch dimension explicit even for batch=1.
    pattern = pattern.reshape(batch, 1, 1, 1)
    ys = jnp.arange(height)[None, :, None, None] / height
    xs = jnp.arange(width)[None, None, :, None] / width
    span_y = 0.25 + 0.5 * random[:, 2]
    span_x = 0.25 + 0.5 * random[:, 3]
    y0 = random[:, 4] * (1 - span_y)
    x0 = random[:, 5] * (1 - span_x)
    rectangle = (ys >= y0) & (ys < y0 + span_y) & (xs >= x0) & (xs < x0 + span_x)
    side = jnp.floor(random[:, 6] * 4)
    band = jnp.where(
        side < 1,
        ys < span_y,
        jnp.where(side < 2, ys >= 1 - span_y, jnp.where(side < 3, xs < span_x, xs >= 1 - span_x)),
    )
    region = jnp.where(random[:, 7] < 0.5, rectangle, band)
    noise = jnp.where(pattern == 1, jnp.where(band, base, 0.0), base)
    noise = jnp.where(pattern == 2, jnp.where(rectangle, base, 0.0), noise)
    noise = jnp.where((pattern == 3) & region & (clean < 1.0), 0.0, noise)
    return jnp.broadcast_to(
        jnp.where(random[:, 8] < config.identity_probability, 0.0, noise), clean.shape
    )


def training_inputs(clean, key, config, matching, *, levels):
    map_key, noise_key = jax.random.split(key)
    noise_levels = pyramid(noise_map(clean, map_key, config), levels)
    targets = pyramid(clean, levels)
    noise_keys = jax.random.split(noise_key, levels)
    noisy = []
    for target, level, k in zip(targets, noise_levels, noise_keys, strict=True):
        count = matching.pool_size if matching.method == "exact" else target.shape[0]
        noise = jax.random.normal(k, (count, *target.shape[1:]))
        if matching.method == "exact":
            noise = match_noise(
                target, level, noise, resolution=matching.cost_resolution, solver=matching.solver
            )
        noisy.append(target * jnp.sqrt(1 - level) + noise * jnp.sqrt(level))
    return TrainingInputs(targets, noise_levels, tuple(noisy))


def sample(model, clean, key, *, strength, steps, levels):
    """Resize first, then initialize and evolve independent noise at each scale.

    strength may be a scalar or a per-pixel map, as in the Krita runtime.
    """
    noise_level = jnp.broadcast_to(jnp.asarray(strength, dtype=clean.dtype), clean.shape)
    noise_levels = pyramid(noise_level, levels)
    targets = pyramid(clean, levels)
    noise_keys = (key,) if levels == 1 else jax.random.split(key, levels)
    images = tuple(
        target * jnp.sqrt(1 - level) + jax.random.normal(k, target.shape) * jnp.sqrt(level)
        for target, level, k in zip(targets, noise_levels, noise_keys, strict=True)
    )
    for step in range(steps):
        current, following = (steps - step) / steps, (steps - step - 1) / steps
        predictions = recursive_forward(
            model, images, tuple(level * current for level in noise_levels)
        )
        ratio = jnp.sqrt(following / current)
        images = tuple(
            image * ratio
            + prediction * (jnp.sqrt(1 - level * following) - jnp.sqrt(1 - level * current) * ratio)
            for image, prediction, level in zip(images, predictions, noise_levels, strict=True)
        )
    return images[0]
