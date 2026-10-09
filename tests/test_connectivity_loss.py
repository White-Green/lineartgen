import heapq

import jax
import jax.numpy as jnp
import numpy as np

from lineartgen.training.connectivity import connection_capacity, connectivity_loss


def oracle(alpha, source, sink):
    best = np.full(alpha.shape, -1.0)
    queue = []
    for y, x in np.argwhere(source):
        best[y, x] = alpha[y, x]
        heapq.heappush(queue, (-float(alpha[y, x]), int(y), int(x)))
    while queue:
        negative, y, x = heapq.heappop(queue)
        if -negative < best[y, x]:
            continue
        if sink[y, x]:
            return -negative
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = y + dy, x + dx
                if 0 <= yy < alpha.shape[0] and 0 <= xx < alpha.shape[1]:
                    candidate = min(-negative, float(alpha[yy, xx]))
                    if candidate > best[yy, xx]:
                        best[yy, xx] = candidate
                        heapq.heappush(queue, (-candidate, yy, xx))
    return 0.0


def anchors(shape):
    a, b = (np.zeros(shape, bool) for _ in range(2))
    a[shape[0] // 2, 0] = True
    b[shape[0] // 2, -1] = True
    return a, b


def test_matches_independent_widest_path_search_and_gradient():
    rng = np.random.default_rng(20261009)
    a, b = anchors((9, 11))
    run = jax.jit(jax.value_and_grad(connection_capacity))
    for _ in range(8):
        alpha = rng.uniform(0.01, 0.99, (9, 11)).astype(np.float32)
        value, gradient = run(alpha, a, b)
        assert float(value) == oracle(alpha, a, b)
        assert np.count_nonzero(gradient) == 1
        y, x = np.argwhere(gradient)[0]
        delta = 1e-4
        changed = alpha.copy()
        changed[y, x] += delta
        derivative = (oracle(changed, a, b) - float(value)) / delta
        np.testing.assert_allclose(derivative, 1, atol=0.002)


def test_gray_bridge_and_antialiased_line_controls():
    a, b = anchors((9, 15))
    target = np.zeros((9, 15), np.float32)
    target[4] = 0.6
    cut = target.copy()
    cut[:, 7] = 0
    run = jax.jit(connection_capacity)
    assert float(run(target, a, b)) == np.float32(0.6)
    assert float(run(cut, a, b)) == 0
    for level in (0.1, 0.25, 0.5):
        bridge = cut.copy()
        bridge[4, 7] = level
        assert float(run(bridge, a, b)) == np.float32(level)

    def loss(x, y, active):
        return connectivity_loss(x[None], y[None], a[None], b[None], active[None])[0]

    value, grad = jax.jit(jax.value_and_grad(loss))(target, target, np.ones_like(a))
    assert value == 0 and np.count_nonzero(grad) == 0
    _, grad = jax.jit(jax.value_and_grad(loss))(bridge, cut, np.ones_like(a))
    assert grad[4, 7] > 0 and np.count_nonzero(grad) == 1
    active = np.ones_like(a)
    active[4, 7] = False
    _, grad = jax.jit(jax.value_and_grad(loss))(bridge, cut, active)
    assert np.count_nonzero(grad) == 0
    # Extra darkness of a connected teacher is handled by pixel/profile losses.
    darker = target * 1.5
    assert loss(darker, target, np.ones_like(a)) == 0


def test_detour_no_boundary_wrap_ties_and_empty_queries():
    alpha = np.zeros((7, 9), np.float32)
    a, b = anchors(alpha.shape)
    alpha[3, [0, -1]] = 1
    assert connection_capacity(jnp.asarray(alpha), a, b) == 0
    alpha[1, :] = 0.4
    alpha[1:4, [0, -1]] = 0.8
    assert connection_capacity(jnp.asarray(alpha), a, b) == np.float32(0.4)
    value, gradient = jax.value_and_grad(connection_capacity)(jnp.asarray(alpha), a, b)
    assert np.isfinite(gradient).all() and gradient.sum() == 1
    selected = np.argwhere(gradient)[0]
    assert alpha[tuple(selected)] == value
    assert connection_capacity(jnp.asarray(alpha), np.zeros_like(a), b) == 0
