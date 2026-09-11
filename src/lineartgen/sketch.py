"""Paired sketch corruption: opacity, blur, width and locally displaced extra strokes.

All spatial operations act on ink (white = zero) and are sampled once before
the recursion pyramid. The target image is never passed through these operations.
Random keys for each component are fixed across the cumulative ablations.
"""

import math

import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

from lineartgen.config import SketchAugmentationConfig


def gaussian_blur(ink, sigma, maximum_sigma):
    radius = math.ceil(3 * maximum_sigma)
    if radius == 0:
        return ink
    axis = jnp.arange(-radius, radius + 1, dtype=ink.dtype)

    def apply(image, scale):
        weights = jnp.exp(-0.5 * jnp.square(axis / jnp.maximum(scale, 1e-3)))
        weights /= weights.sum()
        for kernel in (weights[:, None, None, None], weights[None, :, None, None]):
            image = jax.lax.conv_general_dilated(
                image[None], kernel, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
            )[0]
        return image

    return jax.vmap(apply)(ink, sigma)


def displaced_strokes(ink, key, maximum_radius, displacement):
    batch, height, width, _ = ink.shape
    radii = jax.random.randint(jax.random.fold_in(key, 1), (batch, 1, 1, 1), 0, maximum_radius + 1)
    widened = ink
    for radius in range(1, maximum_radius + 1):
        dilated = jax.lax.reduce_window(
            ink, 0.0, jax.lax.max, (1, 2 * radius + 1, 2 * radius + 1, 1), (1, 1, 1, 1), "SAME"
        )
        widened = jnp.where(radii == radius, dilated, widened)
    yy, xx = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="ij")
    strengths = jax.random.uniform(
        jax.random.fold_in(key, 2), (3, batch, 1, 1, 1), minval=0.35, maxval=1.0
    )
    result = strengths[0] * widened
    for index in range(2):
        flow = jax.random.uniform(
            jax.random.fold_in(key, 10 + index), (batch, 8, 8, 2), minval=-1, maxval=1
        )
        flow = jax.image.resize(flow, (batch, height, width, 2), method="linear")
        offset = jax.random.uniform(
            jax.random.fold_in(key, 20 + index), (batch, 1, 1, 2), minval=-1, maxval=1
        )
        flow = displacement * (flow + offset) / 2

        def warp(image, shift):
            coordinates = jnp.stack((yy + shift[..., 0], xx + shift[..., 1]))
            return map_coordinates(image[..., 0], coordinates, order=1, mode="constant", cval=0)[
                ..., None
            ]

        stroke = strengths[index + 1] * jax.vmap(warp)(widened, flow)
        result = 1 - (1 - result) * (1 - stroke)
    return result


def augment_sketch(clean, key, config: SketchAugmentationConfig):
    if config.mode == "none" or config.probability == 0:
        return clean
    batch = clean.shape[0]
    enabled = jax.random.bernoulli(jax.random.fold_in(key, 0), config.probability, (batch, 1, 1, 1))
    opacity = jax.random.uniform(
        jax.random.fold_in(key, 1),
        (batch, 1, 1, 1),
        minval=config.opacity_min,
        maxval=config.opacity_max,
    )
    ink = jnp.clip((1 - clean) / 2, 0, 1)
    if config.mode == "strokes":
        ink = displaced_strokes(
            ink, jax.random.fold_in(key, 3), config.dilation_radius, config.displacement
        )
    if config.mode in ("blur", "strokes"):
        sigma = jax.random.uniform(
            jax.random.fold_in(key, 2),
            (batch,),
            minval=config.blur_sigma_min,
            maxval=config.blur_sigma_max,
        )
        ink = gaussian_blur(ink, sigma, config.blur_sigma_max)
    source = 1 - 2 * jnp.clip(ink, 0, 1) * opacity
    return jnp.where(enabled, source, clean)


def restoration_target(clean, source, strength):
    """An edit cleans the selected pixels and preserves the actual input elsewhere."""
    return jnp.where(strength > 0, clean, source)
