"""Matched sparse geometry objectives with different attraction reaches."""

from attraction_objective import stage_loss
from flax import nnx
from learned_centerline_model import forward


def objective(model, prepared, targets, base_radius):
    heads = forward(model, prepared)
    weights = [0.5**i for i in range(len(heads))]
    parts = [
        stage_loss(h, t, i, sparse=True, base_radius=base_radius)
        for i, (h, t) in enumerate(zip(heads, targets, strict=True))
    ]
    totals = {
        k: sum(w * p[k] for w, p in zip(weights, parts, strict=True)) / sum(weights)
        for k in parts[0]
    }
    totals["loss"] = sum(totals.values())
    return totals["loss"], totals


@nnx.jit
def wide_gradient(model, prepared, targets):
    def loss(m, p, t):
        return objective(m, p, t, 4.0)

    (_, parts), grads = nnx.value_and_grad(loss, has_aux=True)(model, prepared, targets)
    return parts, grads


@nnx.jit
def narrow_gradient(model, prepared, targets):
    def loss(m, p, t):
        return objective(m, p, t, 1.0)

    (_, parts), grads = nnx.value_and_grad(loss, has_aux=True)(model, prepared, targets)
    return parts, grads
