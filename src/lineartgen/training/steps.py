"""JIT-compiled numerical steps. Filesystem, logging and data loading stay outside JIT."""

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from lineartgen.diffusion import recursive_forward

from .losses import image_loss


def make_steps(config):
    def objective(model, prepared):
        targets, noise_levels, noisy = prepared
        predictions = recursive_forward(model, noisy, noise_levels)
        total, denominator = jnp.zeros(()), 0.0
        metrics = None
        for level, (prediction, target, noise) in enumerate(
            zip(predictions, targets, noise_levels, strict=True)
        ):
            weight = config.diffusion.scale_loss_multiplier**level
            if weight == 0:
                continue  # Still backpropagate through this scale's recursive prediction.
            loss, parts = image_loss(
                prediction, target, noise, config.loss, config.diffusion.min_noise
            )
            total += weight * loss
            denominator += weight
            if metrics is None:
                metrics = parts
        metrics["loss"] = total / denominator
        return metrics["loss"], metrics

    @nnx.jit
    def gradients(model, prepared):
        (_, metrics), grads = nnx.value_and_grad(objective, has_aux=True)(model, prepared)
        return metrics, grads

    @nnx.jit
    def update(model, optimizer, gradients, ema):
        norm = optax.tree.norm(gradients)
        optimizer.update(model, gradients)
        state = nnx.state(model, nnx.Param)
        ema = jax.tree.map(
            lambda old, new: old * config.train.ema_decay + new * (1 - config.train.ema_decay),
            ema,
            state,
        )
        return norm, ema

    return gradients, update, nnx.jit(lambda model, prepared: objective(model, prepared)[1])


def create_optimizer(model, config):
    train = config.train
    peak = train.peak_learning_rate
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=peak / 10 if train.warmup_steps else peak,
        peak_value=peak,
        warmup_steps=train.warmup_steps,
        decay_steps=train.steps,
        end_value=train.minimum_learning_rate,
    )
    optimizer = nnx.Optimizer(
        model,
        optax.chain(
            optax.clip_by_global_norm(train.gradient_clip),
            optax.adamw(schedule, weight_decay=train.weight_decay),
        ),
        wrt=nnx.Param,
    )
    return optimizer, schedule
