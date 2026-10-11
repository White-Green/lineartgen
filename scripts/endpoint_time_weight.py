"""Change only time weighting of the existing bounded endpoint training risk."""

from functools import partial

import jax.numpy as jnp
from anchored_endpoint import endpoints
from flax import nnx
from interval_velocity_fit import interval_error


def objective(model, p, floor):
    raw = endpoints(model, p["images"], p["levels"], p["guides"], p["progress"], False)[0][0]
    error = interval_error(raw, p["target"], p["target"])
    active = p["levels"][0] > 0
    denom = jnp.maximum(active.sum((1, 2, 3)), 1)
    endpoint_mse = ((error**2 * active).sum((1, 2, 3)) / denom).mean()
    scaled = error / jnp.maximum(p["progress"][0], floor)
    loss = ((scaled**2 * active).sum((1, 2, 3)) / denom).mean()
    return loss, dict(loss=loss, interval_endpoint_mse=endpoint_mse)


@partial(nnx.jit, static_argnames=("floor",))
def gradient(model, p, floor):
    (_, metrics), grads = nnx.value_and_grad(lambda m: objective(m, p, floor), has_aux=True)(model)
    return metrics, grads


cap_gradient = partial(gradient, floor=0.25)
endpoint_gradient = partial(gradient, floor=1.0)
