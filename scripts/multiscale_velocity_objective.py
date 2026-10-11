"""Per-scale velocity targets from independently noised resized inputs."""

import jax.numpy as jnp

from lineartgen.diffusion import pyramid


def multiscale_inputs(prepared):
    from velocity_bridge_toy import velocity_inputs

    p = velocity_inputs(prepared)
    p["targets"] = pyramid(p["target"], len(p["images"]))
    p["velocity_targets"] = tuple(
        y - z for y, z in zip(p["targets"], prepared["direct"]["images"], strict=True)
    )
    return p


def stage_velocity_mse(prediction, target, level):
    active = level > 0
    return (
        (jnp.square(prediction - target) * active).sum((1, 2, 3))
        / jnp.maximum(active.sum((1, 2, 3)), 1)
    ).mean()


def multiscale_objective(model, prepared, multiplier=1.0):
    from velocity_bridge_toy import objective, velocity

    # Keep the historical highest-stage objective bitwise identical at weight0.
    if multiplier == 0:
        return objective(model, prepared)
    predictions = velocity(
        model, prepared["images"], prepared["levels"], prepared["guides"], prepared["progress"]
    )
    losses = tuple(
        stage_velocity_mse(p, y, t)
        for p, y, t in zip(
            predictions, prepared["velocity_targets"], prepared["levels"], strict=True
        )
    )
    weights = tuple(multiplier**i for i in range(len(losses)))
    loss = sum(w * value for w, value in zip(weights, losses, strict=True)) / sum(weights)
    return loss, {"loss": loss} | {f"velocity_mse_scale{i}": v for i, v in enumerate(losses)}
