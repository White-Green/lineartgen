"""Supervise endpoints on teacher interpolation states at sampled remaining times."""

from functools import partial

import jax
import jax.numpy as jnp
from anchored_endpoint import MIN_REMAINING, endpoints
from flax import nnx
from interval_velocity_fit import interval_error

from lineartgen.training.losses import image_loss


def objective(model, p, loss_config, min_noise, coefficient=0.1):
    raw = endpoints(model, p["images"], p["levels"], p["guides"], p["progress"])[0][0]
    active = p["levels"][0] > 0
    error = interval_error(raw, p["target"], p["target"])
    weighted = error / jnp.maximum(p["progress"][0], MIN_REMAINING)
    denom = jnp.maximum(active.sum((1, 2, 3)), 1)
    base = ((weighted**2 * active).sum((1, 2, 3)) / denom).mean()

    def single(pred, y, strength):
        return image_loss(
            jnp.clip(pred[None], -1, 1),
            y[None],
            strength[None],
            loss_config,
            min_noise,
            preclip=pred[None],
        )

    values, parts = jax.vmap(single)(raw, p["target"], p["levels"][0])
    auxiliary = values.mean()
    loss = base + coefficient * auxiliary
    return loss, {f"image_{k}": v.mean() for k, v in parts.items()} | dict(
        loss=loss,
        flow_loss=base,
        image_loss=auxiliary,
        weighted_image_loss=coefficient * auxiliary,
        interval_endpoint_mse=((error**2 * active).sum((1, 2, 3)) / denom).mean(),
    )


@partial(nnx.jit, static_argnames=("coefficient", "loss_config", "min_noise"))
def gradient(model, p, loss_config, min_noise, coefficient=0.1):
    (_, metrics), grads = nnx.value_and_grad(
        lambda m: objective(m, p, loss_config, min_noise, coefficient), has_aux=True
    )(model)
    return metrics, grads
