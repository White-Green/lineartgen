"""Apply the existing line appearance loss to a differentiable two-step rollout."""

from functools import partial

import jax
import jax.numpy as jnp
from anchored_endpoint import endpoints
from flax import nnx
from interval_velocity_fit import interval_error
from rollout_endpoint import midpoint
from rollout_endpoint import objective as original_objective

from lineartgen.training.losses import image_loss


def objective(model, p, loss_config, min_noise, coefficient):
    if coefficient == 0:
        return original_objective(model, p, False)
    u0 = tuple(jnp.ones_like(x) for x in p["images"])
    raw0, y0 = endpoints(model, p["images"], p["levels"], p["guides"], u0)
    middle = midpoint(p, y0, teacher=False)
    u1 = tuple(jnp.full_like(x, 0.5) for x in middle)
    raw1, _ = endpoints(model, middle, p["levels"], p["guides"], u1)
    active = p["levels"][0] > 0
    denom = jnp.maximum(active.sum((1, 2, 3)), 1)
    risks = []
    for raw in (raw0[0], raw1[0]):
        err = interval_error(raw, p["target"], p["target"])
        risks.append(((err**2 * active).sum((1, 2, 3)) / denom).mean())
    base = 0.5 * (risks[0] + 4 * risks[1])

    def single(raw, y, t):
        return image_loss(
            jnp.clip(raw[None], -1, 1), y[None], t[None], loss_config, min_noise, preclip=raw[None]
        )

    values, parts = jax.vmap(single)(raw1[0], p["target"], p["levels"][0])
    auxiliary = values.mean()
    loss = base + coefficient * auxiliary
    return loss, {f"image_{k}": v.mean() for k, v in parts.items()} | dict(
        loss=loss,
        initial_mse=risks[0],
        midpoint_mse=risks[1],
        base_loss=base,
        image_loss=auxiliary,
        weighted_image_loss=coefficient * auxiliary,
    )


@partial(nnx.jit, static_argnames=("coefficient", "loss_config", "min_noise"))
def gradient(model, p, loss_config, min_noise, coefficient):
    (_, metrics), grads = nnx.value_and_grad(
        lambda m: objective(m, p, loss_config, min_noise, coefficient), has_aux=True
    )(model)
    return metrics, grads
