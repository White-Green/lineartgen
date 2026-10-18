import jax
import jax.numpy as jnp
import numpy as np
from learned_centerline_model import balanced_bce, decode, stage_loss, target


def as_logits(t):
    h = np.zeros_like(t)
    h[..., [0, 4, 5, 6, 7, 8]] = np.where(t[..., [0, 4, 5, 6, 7, 8]] > 0, 20, -20)
    p = np.clip(t[..., 1:3], 1e-6, 1 - 1e-6)
    h[..., 1:3] = np.log(p / (1 - p))
    h[..., 3] = t[..., 3]
    return h


def test_connections_width_and_endpoints():
    s = np.array([[[3.2, 8.3], [12.7, 8.3]]])
    w = np.array([[2.0, 2.0]])
    t, stats = target(s, w, np.empty((0, 4)), 16)
    out, pred = decode(as_logits(t))
    assert pred["components"] == 1 and pred["edges"] == stats["edges"]
    assert pred["isolated"] == 0
    assert out[8, 4:12].max() < -0.95
    assert out[:5].min() == 1


def test_all_four_directions_and_boundary():
    for end in ([12.7, 4.3], [4.3, 12.7], [12.7, 12.7], [0.3, 12.7]):
        t, g = target(np.array([[[4.3, 4.3], end]]), np.ones((1, 2)), np.empty((0, 4)), 16)
        _, p = decode(as_logits(t))
        assert p["components"] == 1 and p["edges"] == g["edges"]


def test_blank_fill_dot_and_pyramid():
    empty = np.empty((0, 2, 2))
    widths = np.empty((0, 2))
    t, _ = target(empty, widths, np.array([[4, 4, 12, 12]]), 16, 1)
    out, p = decode(as_logits(t))
    assert p["vertices"] == 0 and p["fill_cells"] == 16
    assert np.all(out[2:6, 2:6] == -1) and np.all(out[:2] == 1)
    t, _ = target(np.array([[[8.2, 8.2], [8.2, 8.2]]]), np.ones((1, 2)), np.empty((0, 4)), 16)
    _, p = decode(as_logits(t))
    assert p["isolated"] == p["components"] == 1


def test_probability_not_opacity_and_endpoint_gating():
    h = np.full((8, 8, 9), -20, np.float32)
    h[3, 3, :4] = [1, 0, 0, np.log(2.0)]
    low, _ = decode(h)
    h[3, 3, 0] = 10
    high, p = decode(h)
    np.testing.assert_array_equal(low, high)
    h[3, 3, 4] = 10
    _, p = decode(h)
    assert p["edges"] == 0 and p["isolated"] == 1


def test_background_loss_and_parameter_gradients():
    t, _ = target(
        np.array([[[3.2, 8.3], [12.7, 8.3]]]), np.array([[2.0, 2.0]]), np.empty((0, 4)), 16
    )
    h = jnp.zeros((1, 16, 16, 9))
    truth = jnp.asarray(t[None])
    g = jax.grad(lambda a: sum(stage_loss(a, truth).values()))(h)
    assert np.isfinite(g).all()
    assert g[0, 0, 0, 0] > 0 and np.asarray(g[..., 3])[np.asarray(truth[..., 0]) > 0].min() < 0
    blank = jnp.zeros((1, 8, 8))
    grad = jax.grad(lambda a: balanced_bce(a, blank))(blank)
    assert np.all(grad > 0)
    assert np.all(np.asarray(g[..., 1:4])[np.asarray(truth[..., 0]) == 0] == 0)


def test_halo_retains_outside_centerline_capsule():
    from centerline_grid import encode, render
    from learned_centerline_study import HALO, crop

    s = np.array([[[9.2, -0.9], [14.3, 10.7]]])
    w = np.ones((1, 2)) * 2
    rect = np.empty((0, 4))
    t, _ = target(s + HALO, w, rect, 32 + 2 * HALO)
    decoded, _ = decode(as_logits(t))
    g = encode(s, w, 1)
    expected = render(g["segments"], g["segment_widths"], rect, 32)
    np.testing.assert_allclose(crop(decoded), expected, atol=1e-4, rtol=1e-4)
