"""Forward noise, image pyramids and recursive conditioning; independent of the trainer."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from lineartgen.config import DiffusionConfig
from lineartgen.masks import boundary_rectangle
from lineartgen.noise_matching import match_noise
from lineartgen.sketch import augment_sketch, restoration_target


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


def recursive_forward_all(model, images, noise_levels):
    """All retained subnets share features, with a separate recursive state per prefix."""
    smallest = images[-1]
    count = getattr(model, "num_subnets", 1)
    insert = jnp.zeros(
        (smallest.shape[0], smallest.shape[1] // 2, smallest.shape[2] // 2, count),
        dtype=smallest.dtype,
    )
    outputs = []
    for image, noise in zip(reversed(images), reversed(noise_levels), strict=True):
        insert = model(image, noise, insert)
        outputs.append(insert)
    return tuple(
        tuple(output[..., i : i + 1] for output in reversed(outputs)) for i in range(count)
    )


def recursive_forward(model, images, noise_levels):
    """Return the selected prefix's predictions, including its internal smaller states."""
    return recursive_forward_all(model, images, noise_levels)[-1]


def strength_from_uniform(uniform, config: DiffusionConfig):
    """Inverse CDF for training strengths, leaving all other random draws unchanged."""
    if config.strength_sampling == "uniform":
        return config.min_noise + uniform * (1 - config.min_noise)
    p, split = config.low_strength_probability, config.low_strength_max
    return jnp.where(
        uniform < p,
        config.min_noise + (uniform / p) * (split - config.min_noise),
        split + ((uniform - p) / (1 - p)) * (1 - split),
    )


def noise_map(clean, key, config: DiffusionConfig):
    """Full, band, rectangle and region-with-protected-lines patterns, entirely on device."""
    batch, height, width, _ = clean.shape
    random = jax.random.uniform(key, (batch, 10, 1, 1, 1))
    base = strength_from_uniform(random[:, 0], config)
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
    if config.mask_sampling == "boundary_rect":
        rectangle = boundary_rectangle(clean, rectangle, key, config.boundary_candidates)
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


def training_inputs(clean, key, config, matching, *, levels, sketch=None):
    map_key, noise_key = jax.random.split(key)
    source = clean
    if sketch is not None and sketch.mode != "none":
        if matching.method != "independent":
            raise ValueError("Sketch augmentation currently requires independent Gaussian noise")
        source = augment_sketch(clean, jax.random.fold_in(key, 0x534B4554), sketch)
    strength = noise_map(source, map_key, config)
    noise_levels = pyramid(strength, levels)
    targets = pyramid(restoration_target(clean, source, strength), levels)
    sources = pyramid(source, levels)
    noise_keys = jax.random.split(noise_key, levels)
    noisy = []
    for target, source, level, k in zip(targets, sources, noise_levels, noise_keys, strict=True):
        count = matching.pool_size if matching.method == "exact" else target.shape[0]
        generated = matching.generation_pool_size or count
        if generated < count:
            raise ValueError("generation_pool_size is smaller than the required candidate count")
        noise = jax.random.normal(k, (generated, *target.shape[1:]))[:count]
        if matching.method == "exact":
            noise = match_noise(
                target, level, noise, resolution=matching.cost_resolution, solver=matching.solver
            )
        noisy.append(source * jnp.sqrt(1 - level) + noise * jnp.sqrt(level))
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
    return sample_prepared(model, images, noise_levels, steps=steps)


def sample_prepared(model, images, noise_levels, *, steps):
    """Evolve an already sampled pyramid, permitting paired recursion diagnostics."""
    for step in range(steps):
        current, following = (steps - step) / steps, (steps - step - 1) / steps
        predictions = recursive_forward(
            model, images, tuple(level * current for level in noise_levels)
        )
        ratio = jnp.sqrt(following / current)
        updated = tuple(
            image * ratio
            + prediction * (jnp.sqrt(1 - level * following) - jnp.sqrt(1 - level * current) * ratio)
            for image, prediction, level in zip(images, predictions, noise_levels, strict=True)
        )
        if getattr(model, "output_mode", "direct") == "residual":
            # The model is identity at t=0; avoid roundoff from the sampler's
            # algebraically cancelling terms in those protected pixels as well.
            updated = tuple(
                jnp.where(level == 0, image, value)
                for image, value, level in zip(images, updated, noise_levels, strict=True)
            )
        images = updated
    return images[0]
