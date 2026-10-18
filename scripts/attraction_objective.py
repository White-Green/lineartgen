"""Regress centerline attraction offsets, keeping existence separate from opacity."""

import jax.numpy as jnp
import optax
from attraction_graph import radius
from flax import nnx
from learned_centerline_model import balanced_bce, forward
from sparse_graph_objective import count_bce


def stage_loss(head, truth, level, *, sparse=False, base_radius=4.0):
    occupied = truth[..., 0]
    count = jnp.maximum(occupied.sum((1, 2)), 1)
    offset = optax.huber_loss(
        radius(level, base_radius) * jnp.tanh(head[..., 1:3]), truth[..., 1:3]
    ).mean(-1)
    width = optax.huber_loss(head[..., 3], truth[..., 3])
    bce = count_bce if sparse else balanced_bce
    return dict(
        vertex=bce(head[..., 0], occupied),
        edge=bce(head[..., 4:8], truth[..., 4:8]),
        fill=balanced_bce(head[..., 8], truth[..., 8]),
        position=((offset * occupied).sum((1, 2)) / count).mean(),
        width=((width * occupied).sum((1, 2)) / count).mean(),
    )


def objective(model, prepared, targets, *, sparse=False):
    heads = forward(model, prepared)
    weights = [0.5**i for i in range(len(heads))]
    parts = [
        stage_loss(h, t, i, sparse=sparse)
        for i, (h, t) in enumerate(zip(heads, targets, strict=True))
    ]
    totals = {
        k: sum(w * p[k] for w, p in zip(weights, parts, strict=True)) / sum(weights)
        for k in parts[0]
    }
    totals["loss"] = sum(totals.values())
    return totals["loss"], totals


@nnx.jit
def balanced_gradient(model, prepared, targets):
    def loss(m, p, t):
        return objective(m, p, t, sparse=False)

    (_, parts), grads = nnx.value_and_grad(loss, has_aux=True)(model, prepared, targets)
    return parts, grads


@nnx.jit
def sparse_gradient(model, prepared, targets):
    def loss(m, p, t):
        return objective(m, p, t, sparse=True)

    (_, parts), grads = nnx.value_and_grad(loss, has_aux=True)(model, prepared, targets)
    return parts, grads
