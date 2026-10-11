"""Point velocity supervision plus an image objective only at the starting atom."""

import jax
import jax.numpy as jnp
from velocity_bridge_toy import objective, recursive_velocity

from lineartgen.training.losses import image_loss

COEFFICIENT = 0.1


def start_image_loss(physical, prepared, loss_config, min_noise):
    def single(p, y, t):
        return image_loss(
            jnp.clip(p[None], -1, 1),
            y[None],
            t[None],
            loss_config,
            min_noise,
            preclip=p[None],
        )

    values, parts = jax.vmap(single)(physical, prepared["target"], prepared["levels"][0])
    at_start = prepared["progress"][0][:, 0, 0, 0] == 1
    # Batch mean, not mean over the selected subset: the atom occurs 25% of the time.
    return (
        jnp.mean(jnp.where(at_start, values, 0)),
        at_start.mean(),
        {f"start_{name}": jnp.mean(jnp.where(at_start, value, 0)) for name, value in parts.items()},
    )


def endpoint_image_objective(model, prepared, loss_config, min_noise, coefficient=COEFFICIENT):
    if coefficient == 0:
        return objective(model, prepared)
    velocities, endpoints = recursive_velocity(
        model, prepared["images"], prepared["levels"], prepared["guides"], prepared["progress"]
    )
    active = prepared["levels"][0] > 0
    error = jnp.square(velocities[0] - prepared["velocity_target"])
    mse = ((error * active).sum((1, 2, 3)) / jnp.maximum(active.sum((1, 2, 3)), 1)).mean()
    auxiliary, fraction, parts = start_image_loss(endpoints[0], prepared, loss_config, min_noise)
    loss = mse + coefficient * auxiliary
    return loss, parts | dict(
        loss=loss,
        velocity_mse=mse,
        start_image_loss=auxiliary,
        start_fraction=fraction,
        weighted_start_image=coefficient * auxiliary,
    )
