"""Supervise two endpoint predictions on teacher or differentiable rollout states."""

from functools import partial

import jax.numpy as jnp
from anchored_endpoint import endpoints
from flax import nnx
from interval_velocity_fit import interval_error

from lineartgen.diffusion import pyramid


def inputs(prepared):
    return dict(prepared["direct"])


def midpoint(p, predicted, *, teacher):
    clean = pyramid(p["target"], len(p["images"])) if teacher else predicted
    return tuple(
        jnp.where(t > 0, 0.5 * x + 0.5 * y, s)
        for x, y, t, s in zip(p["images"], clean, p["levels"], p["guides"], strict=True)
    )


def objective(model, p, teacher):
    progress0 = tuple(jnp.ones_like(x) for x in p["images"])
    raw0, y0 = endpoints(model, p["images"], p["levels"], p["guides"], progress0)
    middle = midpoint(p, y0, teacher=teacher)
    progress1 = tuple(jnp.full_like(x, 0.5) for x in middle)
    raw1, _ = endpoints(model, middle, p["levels"], p["guides"], progress1)
    active = p["levels"][0] > 0
    denom = jnp.maximum(active.sum((1, 2, 3)), 1)
    risks = []
    for raw in (raw0[0], raw1[0]):
        err = interval_error(raw, p["target"], p["target"])
        risks.append(((err**2 * active).sum((1, 2, 3)) / denom).mean())
    loss = 0.5 * (risks[0] + 4 * risks[1])
    return loss, dict(loss=loss, initial_mse=risks[0], midpoint_mse=risks[1])


@partial(nnx.jit, static_argnames=("teacher",))
def gradient(model, p, teacher):
    (_, metrics), grads = nnx.value_and_grad(lambda m: objective(m, p, teacher), has_aux=True)(
        model
    )
    return metrics, grads


teacher_gradient = partial(gradient, teacher=True)
rollout_gradient = partial(gradient, teacher=False)
