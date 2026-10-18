"""Binary fill objective with a per-image hard-negative tail."""

import math

import jax
import jax.numpy as jnp
import optax


def hard_negative_bce(logits, labels, fraction=0.01):
    """Keep the positive mean; average the largest negative BCE values.

    K uses the full spatial size, so it stays static under JIT. If there
    are fewer than K negative cells, average only the available negatives.
    Labels must be binary, as in the graph's fill targets.
    """
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if logits.shape != labels.shape or logits.ndim < 2:
        raise ValueError("matching batch and spatial shapes required")
    loss = optax.sigmoid_binary_cross_entropy(logits, labels)
    loss, labels = loss.reshape(loss.shape[0], -1), labels.reshape(labels.shape[0], -1)
    positive = (loss * labels).sum(-1) / jnp.maximum(labels.sum(-1), 1)
    negative = labels == 0
    k = max(1, math.ceil(loss.shape[-1] * fraction))
    # ROCm's large-K top_k exceeds shared-memory limits at 320x320 / K=1024.
    # Stable descending sort has the same values and tie order as top_k.
    candidates = jnp.where(negative, loss, -jnp.inf)
    indices = jnp.broadcast_to(jnp.arange(loss.shape[-1]), loss.shape)
    _, order = jax.lax.sort_key_val(
        -jax.lax.stop_gradient(candidates), indices, dimension=-1, is_stable=True
    )
    # Differentiate the selected original BCE values, not the sort operation.
    values = jnp.take_along_axis(candidates, order[:, :k], axis=-1)
    total = jnp.where(jnp.isfinite(values), values, 0).sum(-1)
    denominator = jnp.maximum(jnp.minimum(negative.sum(-1), k), 1)
    return (positive + total / denominator).mean()
