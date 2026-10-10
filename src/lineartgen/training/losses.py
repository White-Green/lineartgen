"""Loss choices are kept here so experiments do not require editing the trainer."""

import jax
import jax.numpy as jnp

from .boundary import boundary_distance_loss
from .cldice import centerline_loss
from .ink_structure import ink_structure_losses
from .interval import interval_absolute_error, interval_squared_error
from .lineart import line_region_loss, profile_loss
from .opacity import opacity_deficit_loss
from .paper import paper_peak_loss


def region_mean(error, weights, region):
    weights = weights * region
    return (error * weights).sum() / jnp.maximum(weights.sum(), 1e-6)


def tone_weights(clean, power=1.0, local_radius=0):
    if power == 0:
        return jnp.ones_like(clean)
    black = jnp.clip((1 - clean) * 0.5, 0, 1)
    white = 1 - black
    axes = (1, 2)
    weight = black / (2 * black.mean(axes, keepdims=True) + 1e-6)
    weight += white / (2 * white.mean(axes, keepdims=True) + 1e-6)
    weight = weight / (weight.mean(axes, keepdims=True) + 1e-6)
    if power != 1:
        weight = jnp.power(weight, power)
        weight = weight / weight.mean(axes, keepdims=True)
    if local_radius:
        window = 2 * local_radius + 1

        def local_sum(x):
            x = jax.lax.reduce_window(x, 0.0, jax.lax.add, (1, 1, window, 1), (1, 1, 1, 1), "SAME")
            return jax.lax.reduce_window(
                x, 0.0, jax.lax.add, (1, window, 1, 1), (1, 1, 1, 1), "SAME"
            )

        density = local_sum(black) / local_sum(jnp.ones_like(black))
        local = black / (2 * density + 1e-6)
        local += white / (2 * (1 - density) + 1e-6)
        weight *= jnp.power(local, power)
        weight /= weight.mean(1, keepdims=True).mean(2, keepdims=True)
    return weight


def absolute_error(prediction, target):
    """Absolute error with the zero subgradient at an exact match."""
    delta = prediction - target
    return jnp.where(delta == 0, jnp.zeros_like(delta), jnp.abs(delta))


def sobel(image):
    kernel = jnp.array(
        [
            [[-0.25, -0.25], [0, -0.5], [0.25, -0.25]],
            [[-0.5, 0], [0, 0], [0.5, 0]],
            [[-0.25, 0.25], [0, 0.5], [0.25, 0.25]],
        ],
        dtype=image.dtype,
    ).reshape(3, 3, 1, 2)
    return jax.lax.conv_general_dilated(
        image, kernel, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
    )


def image_loss(
    prediction, clean, noise_level, config, min_noise, *, profiles=None, level=0, preclip=None
):
    active = noise_level > min_noise
    weights = (
        tone_weights(clean, config.tone_balance_power, config.local_tone_radius)
        if config.balance_tones
        else jnp.ones_like(clean)
    )
    if config.pixel_loss in ("interval_mse", "interval_l1"):
        if preclip is None:
            raise ValueError(f"{config.pixel_loss} requires the physical pre-clip prediction")
        error = (
            interval_absolute_error(preclip, clean)
            if config.pixel_loss == "interval_l1"
            else interval_squared_error(preclip, clean)
        )
    elif config.pixel_loss == "clipped_l1":
        error = absolute_error(prediction, clean)
    else:
        error = jnp.square(prediction - clean)
    floor = max(min_noise, config.noise_weight_floor)
    denoise = region_mean(error / jnp.maximum(noise_level, floor), weights, active)
    identity = region_mean(error, weights, ~active)
    black = jnp.clip((1 - prediction) / 2, 0, 1)
    edge = (
        jnp.mean(jnp.square(sobel(black) - sobel((1 - clean) / 2)))
        if config.edge_weight
        else jnp.zeros(())
    )
    contrast_error = 4 * black * (1 - black)
    contrast = (
        region_mean(contrast_error, jnp.ones_like(clean), active)
        if config.contrast_active_only
        else jnp.mean(contrast_error)
    )
    loss = denoise + config.identity_weight * identity
    loss += config.edge_weight * edge + config.contrast_weight * contrast
    parts = {"denoise": denoise, "identity": identity, "edge": edge, "contrast": contrast}
    if config.opacity_deficit_weight:
        if config.opacity_deficit_preclip and preclip is None:
            raise ValueError("Preclip opacity requires the physical pre-clip prediction")
        opacity_deficit = opacity_deficit_loss(
            preclip if config.opacity_deficit_preclip else prediction,
            clean,
            active,
            unclipped=config.opacity_deficit_preclip,
        )
        loss += config.opacity_deficit_weight * opacity_deficit
        parts["opacity_deficit"] = opacity_deficit
    if config.paper_peak_weight:
        paper = paper_peak_loss(prediction, clean, active, tile=config.paper_peak_tile)
        loss += config.paper_peak_weight * paper
        parts["paper_peak"] = paper
    if config.ink_keep_weight or config.neighbor_keep_weight or config.neighbor_paper_weight:
        structure = ink_structure_losses(
            prediction,
            clean,
            active,
            halo_radius=config.line_region_radius,
            neighbor_radius=config.neighbor_radius,
        )
        for name, value in structure.items():
            loss += getattr(config, name + "_weight") * value
        parts.update(structure)
    if config.line_region_weight:
        region = line_region_loss(prediction, clean, active, config.line_region_radius)
        loss += config.line_region_weight * region
        parts["line_region"] = region
    if config.line_profile_weight:
        if profiles is None:
            raise ValueError("line_profile_weight requires generated curve profile coordinates")
        profile = profile_loss(prediction, clean, active, profiles, level=level)
        loss += config.line_profile_weight * profile
        parts["line_profile"] = profile
    if config.boundary_weight:
        boundary = boundary_distance_loss(prediction, clean, active, config.boundary_radius)
        loss += config.boundary_weight * boundary
        parts["boundary"] = boundary
    if config.cldice_weight:
        centerline = centerline_loss(prediction, clean, active, config.cldice_iterations)
        loss += config.cldice_weight * centerline
        parts["centerline"] = centerline
    return loss, parts
