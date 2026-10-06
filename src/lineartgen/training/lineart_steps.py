"""Single-generator loss study, preserving recursion, accumulation and generator optimizer."""

import jax
import jax.numpy as jnp
from flax import nnx

from lineartgen.diffusion import recursive_forward

from .lineart_adversarial import discriminator_loss, generator_prior
from .losses import image_loss
from .steps import _add_trees, _average_and_norm


def objective(model, prepared, profiles, config):
    predictions = recursive_forward(model, prepared.noisy, prepared.noise_levels)
    total, denominator, metrics = jnp.zeros(()), 0.0, {}
    for level, (prediction, target, noise) in enumerate(
        zip(predictions, prepared.targets, prepared.noise_levels, strict=True)
    ):
        weight = config.diffusion.scale_loss_multiplier**level
        if weight == 0:
            continue
        loss, parts = image_loss(
            prediction,
            target,
            noise,
            config.loss,
            config.diffusion.min_noise,
            profiles=profiles,
            level=level,
        )
        total += weight * loss
        denominator += weight
        if level == 0:
            metrics = parts
    return total / denominator, metrics, predictions


def make_lineart_steps(config, *, adversarial=False, patch_size=64):
    if config.model.architecture != "unet":
        raise ValueError("This study compares a single UNet; no shared critic across nested widths")
    if adversarial and config.diffusion.scale_loss_multiplier != 0:
        raise ValueError("The patch critic study currently requires top-resolution-only loss")

    def generator_loss(model, prepared, profiles, coordinates, critic, prior_weight):
        total, metrics, predictions = objective(model, prepared, profiles, config)
        if adversarial:
            prior = generator_prior(
                critic,
                predictions[0],
                prepared,
                coordinates,
                patch_size,
                config.diffusion.min_noise,
            )
            total += prior_weight * prior
            metrics = dict(metrics, prior=prior, prior_weight=prior_weight)
        return total, dict(metrics, loss=total)

    @nnx.jit
    def generator_gradient(model, prepared, profiles, coordinates, critic, prior_weight):
        (_, metrics), gradient = nnx.value_and_grad(generator_loss, argnums=0, has_aux=True)(
            model, prepared, profiles, coordinates, critic, prior_weight
        )
        return metrics, gradient

    @nnx.jit
    def critic_gradient(critic, model, prepared, coordinates):
        prediction = jax.lax.stop_gradient(
            recursive_forward(model, prepared.noisy, prepared.noise_levels)[0]
        )
        (_, metrics), gradient = nnx.value_and_grad(discriminator_loss, argnums=0, has_aux=True)(
            critic, prediction, prepared, coordinates, patch_size, config.diffusion.min_noise
        )
        return metrics, gradient

    @nnx.jit
    def critic_update(critic, optimizer, gradient):
        optimizer.update(critic, gradient)

    return generator_gradient, critic_gradient, critic_update


def accumulate_generator(gradient, model, prepared, profiles, coordinates, critic, weight, train):
    accumulated = None
    for i in range(train.accumulate_steps):
        begin, end = i * train.batch_size, (i + 1) * train.batch_size
        part = jax.tree.map(
            lambda x, begin=begin, end=end: x[begin:end], (prepared, profiles, coordinates)
        )
        update = gradient(model, *part, critic, jnp.asarray(weight, jnp.float32))
        accumulated = update if accumulated is None else _add_trees(accumulated, update)
    return _average_and_norm(*accumulated[::-1], train.accumulate_steps)


def accumulate_critic(gradient, critic, model, prepared, coordinates, train):
    accumulated = None
    for i in range(train.accumulate_steps):
        begin, end = i * train.batch_size, (i + 1) * train.batch_size
        part = jax.tree.map(lambda x, begin=begin, end=end: x[begin:end], (prepared, coordinates))
        update = gradient(critic, model, *part)
        accumulated = update if accumulated is None else _add_trees(accumulated, update)
    return _average_and_norm(*accumulated[::-1], train.accumulate_steps)
