"""Keep sparse R1 graph objectives, replace only the fill negative mean."""

from attraction_objective import stage_loss as original_stage_loss
from fill_hard_negative import hard_negative_bce
from flax import nnx
from learned_centerline_model import forward


def stage_loss(head, truth, level):
    parts = original_stage_loss(head, truth, level, sparse=True, base_radius=1.0)
    parts["fill"] = hard_negative_bce(head[..., 8], truth[..., 8])
    return parts


def objective(model, prepared, targets):
    heads = forward(model, prepared)
    weights = [0.5**i for i in range(len(heads))]
    parts = [stage_loss(h, t, i) for i, (h, t) in enumerate(zip(heads, targets, strict=True))]
    totals = {
        k: sum(w * p[k] for w, p in zip(weights, parts, strict=True)) / sum(weights)
        for k in parts[0]
    }
    totals["loss"] = sum(totals.values())
    return totals["loss"], totals


@nnx.jit
def hard_negative_gradient(model, prepared, targets):
    (_, parts), gradients = nnx.value_and_grad(objective, has_aux=True)(model, prepared, targets)
    return parts, gradients
