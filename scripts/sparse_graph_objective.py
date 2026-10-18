"""Classify sparse graph edges without averaging false positives over all paper."""

import optax
from flax import nnx
from learned_centerline_model import forward, stage_loss


def count_bce(logits, labels):
    axes = tuple(range(1, logits.ndim))
    cells = logits[0].size
    normalizer = labels.sum(axes) + 0.001 * cells + 1e-6
    values = optax.sigmoid_binary_cross_entropy(logits, labels)
    return (values.sum(axes) / normalizer).mean()


def sparse_stage_loss(head, truth):
    parts = stage_loss(head, truth)
    parts["vertex"] = count_bce(head[..., 0], truth[..., 0])
    parts["edge"] = count_bce(head[..., 4:8], truth[..., 4:8])
    return parts


def objective(model, prepared, targets):
    heads = forward(model, prepared)
    weights = [0.5**i for i in range(len(heads))]
    parts = [sparse_stage_loss(h, t) for h, t in zip(heads, targets, strict=True)]
    totals = {
        k: sum(w * p[k] for w, p in zip(weights, parts, strict=True)) / sum(weights)
        for k in parts[0]
    }
    totals["loss"] = sum(totals.values())
    return totals["loss"], totals


@nnx.jit
def gradient(model, prepared, targets):
    (_, parts), grads = nnx.value_and_grad(objective, has_aux=True)(model, prepared, targets)
    return parts, grads
