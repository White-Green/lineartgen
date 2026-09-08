"""Loss choices are kept here so experiments do not require editing the trainer."""

import jax
import jax.numpy as jnp


def region_mean(error, weights, region):
    weights = weights * region
    return (error * weights).sum() / jnp.maximum(weights.sum(), 1e-6)


def tone_weights(clean):
    black = jnp.clip((1 - clean) * 0.5, 0, 1)
    white = 1 - black
    axes = (1, 2)
    weight = black / (2 * black.mean(axes, keepdims=True) + 1e-6)
    weight += white / (2 * white.mean(axes, keepdims=True) + 1e-6)
    return weight / (weight.mean(axes, keepdims=True) + 1e-6)


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


def image_loss(prediction, clean, noise_level, config, min_noise):
    active = noise_level > min_noise
    weights = tone_weights(clean) if config.balance_tones else jnp.ones_like(clean)
    error = jnp.square(prediction - clean)
    denoise = region_mean(error / jnp.maximum(noise_level, min_noise), weights, active)
    identity = region_mean(error, weights, ~active)
    black = jnp.clip((1 - prediction) / 2, 0, 1)
    edge = (
        jnp.mean(jnp.square(sobel(black) - sobel((1 - clean) / 2)))
        if config.edge_weight
        else jnp.zeros(())
    )
    contrast = jnp.mean(4 * black * (1 - black))
    loss = denoise + config.identity_weight * identity
    loss += config.edge_weight * edge + config.contrast_weight * contrast
    return loss, {"denoise": denoise, "identity": identity, "edge": edge, "contrast": contrast}
