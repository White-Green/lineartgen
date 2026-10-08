"""JIT-compiled numerical steps. Filesystem, logging and data loading stay outside JIT."""

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from lineartgen.diffusion import recursive_forward_all, recursive_forward_with_preclip

from .losses import image_loss


@jax.jit
def _add_trees(a, b):
    return jax.tree.map(lambda x, y: x + y, a, b)


@jax.jit
def _average_and_norm(gradients, metrics, count):
    gradients = jax.tree.map(lambda x: x / count, gradients)
    metrics = jax.tree.map(lambda x: x / count, metrics)
    return metrics, gradients, optax.tree.norm(gradients)


def accumulate_gradients(gradient_step, model, prepared, train):
    """One full optimizer batch; fuse tree arithmetic instead of dispatching per tensor."""
    accumulated = None
    for micro in range(train.accumulate_steps):
        start = micro * train.batch_size
        batch = jax.tree.map(lambda x, start=start: x[start : start + train.batch_size], prepared)
        parts = gradient_step(model, batch)
        accumulated = parts if accumulated is None else _add_trees(accumulated, parts)
    metrics, gradients = accumulated
    return _average_and_norm(gradients, metrics, train.accumulate_steps)


def make_steps(config, *, all_subnets=True):
    def objective(model, prepared):
        targets, noise_levels, noisy = prepared[:3]
        guides = prepared.guides if config.model.source_conditioning else None
        progress = prepared.progress if config.model.progress_conditioning else None
        if config.loss.pixel_loss == "interval_mse":
            predictions, preclip = recursive_forward_with_preclip(
                model, noisy, noise_levels, guides=guides, progress=progress
            )
            subnets = (predictions,)
        else:
            subnets = recursive_forward_all(
                model, noisy, noise_levels, guides=guides, progress=progress
            )
            preclip = (None,) * len(noisy)
        if not all_subnets:
            subnets = (subnets[-1],)
        metrics = {}
        for index, predictions in enumerate(subnets):
            total, denominator = jnp.zeros(()), 0.0
            highest = None
            for level, (prediction, target, noise) in enumerate(
                zip(predictions, targets, noise_levels, strict=True)
            ):
                weight = config.diffusion.scale_loss_multiplier**level
                if weight == 0:
                    continue  # Gradients still traverse this scale's recursive prediction.
                loss, parts = image_loss(
                    prediction,
                    target,
                    noise,
                    config.loss,
                    config.diffusion.min_noise,
                    preclip=preclip[level],
                )
                total += weight * loss
                denominator += weight
                if highest is None:
                    highest = parts
            highest["loss"] = total / denominator
            for name, value in highest.items():
                metrics[name] = metrics.get(name, 0.0) + value / len(subnets)
                if len(subnets) > 1:
                    metrics[f"subnet_{index}/{name}"] = value
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
