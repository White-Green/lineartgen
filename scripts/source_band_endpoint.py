"""Emphasize the observed rough-line region without teacher-dependent weights."""

import jax
import jax.numpy as jnp
from anchored_endpoint import MIN_REMAINING, endpoints
from flax import nnx
from interval_velocity_fit import interval_error


def observed_band(source, levels):
    ink = jnp.clip((1 - source) / 2, 0, 1)
    band = jax.lax.reduce_window(ink, -jnp.inf, jax.lax.max, (1, 9, 9, 1), (1, 1, 1, 1), "SAME")
    return jnp.where(levels > 0, band, 0)


def risk(error_squared, source, levels):
    active = levels > 0
    base = (error_squared * active).sum((1, 2, 3)) / jnp.maximum(active.sum((1, 2, 3)), 1)
    band = observed_band(source, levels)
    mass = band.sum((1, 2, 3))
    local = (error_squared * band).sum((1, 2, 3)) / jnp.maximum(mass, 1e-12)
    local = jnp.where(mass > 0, local, base)
    return 0.5 * (base + local).mean(), base.mean(), local.mean()


def objective(model, p):
    raw = endpoints(model, p["images"], p["levels"], p["guides"], p["progress"], False)[0][0]
    error = interval_error(raw, p["target"], p["target"])
    scaled = error / jnp.maximum(p["progress"][0], MIN_REMAINING)
    loss, base, local = risk(scaled**2, p["guides"][0], p["levels"][0])
    active = p["levels"][0] > 0
    mse = ((error**2 * active).sum((1, 2, 3)) / jnp.maximum(active.sum((1, 2, 3)), 1)).mean()
    return loss, dict(
        loss=loss, interval_endpoint_mse=mse, unweighted_risk=base, source_band_risk=local
    )


@nnx.jit
def band_gradient(model, p):
    (_, metrics), grads = nnx.value_and_grad(objective, has_aux=True)(model, p)
    return metrics, grads
