"""Experimental, query-local maximin opacity supervision.

This is a vertex-capacity adaptation inspired by MALIS, not a topology guarantee.
Queries and patches are training metadata; inference receives no teacher geometry.
The graph search is discrete. Gradients pass only through a selected bottleneck
on a winning path, a piecewise-linear subgradient (ties choose one path).
"""

import jax
import jax.numpy as jnp


def _capacity_and_index(alpha, source, sink):
    height, width = alpha.shape
    index = jnp.arange(height * width, dtype=jnp.int32).reshape(alpha.shape)
    capacity = jnp.where(source, alpha, -1.0)
    bottleneck = jnp.where(source, index, height * width)

    def select(a, b):
        take = (b[0] > a[0]) | ((b[0] == a[0]) & (b[1] < a[1]))
        return jnp.where(take, b[0], a[0]), jnp.where(take, b[1], a[1])

    def advance(state):
        iteration, capacity, bottleneck, _ = state
        best, origin = jax.lax.reduce_window(
            (capacity, bottleneck),
            (jnp.float32(-1), jnp.int32(height * width)),
            select,
            (3, 3),
            (1, 1),
            "SAME",
        )
        candidate = jnp.minimum(alpha, best)
        changed = candidate > capacity
        candidate_index = jnp.where(alpha <= best, index, origin)
        return (
            iteration + 1,
            jnp.where(changed, candidate, capacity),
            jnp.where(changed, candidate_index, bottleneck),
            jnp.any(changed),
        )

    # A simple path has at most V-1 edges. Stop earlier only at a fixed point.
    iterations, capacity, bottleneck, _ = jax.lax.while_loop(
        lambda s: (s[0] < height * width - 1) & s[3],
        advance,
        (jnp.int32(0), capacity, bottleneck, jnp.bool_(True)),
    )
    value = jnp.max(jnp.where(sink, capacity, -1.0))
    critical = jnp.min(jnp.where(sink & (capacity == value), bottleneck, height * width))
    valid = jnp.any(source) & jnp.any(sink)
    return jnp.where(valid, value, 0.0), jnp.minimum(critical, height * width - 1), iterations


def connection_capacity(alpha, source, sink):
    """One HxW patch, boolean anchor neighborhoods; no wrapping at boundaries.

    Alpha must be float32 in [0, 1]. Empty anchors return zero. An exact
    eight-neighbor graph search selects an opacity value from the input.
    """
    _, critical, _ = _capacity_and_index(jax.lax.stop_gradient(alpha), source, sink)
    valid = jnp.any(source) & jnp.any(sink)
    return jnp.where(valid, alpha.reshape(-1)[critical], 0.0)


def connectivity_loss(predicted_alpha, target_alpha, source, sink, active=None):
    """Mean squared capacity error for disconnected queries, deficit for connected.

    Inputs are NxHxW patches. A nonzero teacher capacity is an antialiased
    lower bound, not an instruction to make the line opaque or puncture it.
    Protected pixels participate in paths but receive no gradient. Empty
    anchor queries contribute zero. This alone does not supervise line width
    or require the whole gap to be white.
    """
    if active is not None:
        predicted_alpha = jnp.where(active, predicted_alpha, jax.lax.stop_gradient(predicted_alpha))
    predicted = jax.vmap(connection_capacity)(predicted_alpha, source, sink)
    target = jax.lax.stop_gradient(jax.vmap(connection_capacity)(target_alpha, source, sink))
    error = jnp.where(target > 0, jax.nn.relu(target - predicted), predicted - target)
    return jnp.mean(jnp.square(error)), (predicted, target)
