import jax
import jax.numpy as jnp
import numpy as np
import pytest
from bridge_time_grid import sample_grid, time_nodes
from projected_endpoint import advance, sample
from velocity_bridge_toy import initialize


@pytest.mark.parametrize("steps", [1, 8, 32])
def test_grid_remaining_time_is_positive(steps):
    nodes = time_nodes(steps)
    assert np.all(1 - nodes[:-1] > 0)
    assert np.all(np.diff(nodes) > 0)
    assert nodes[-1] == 1


def test_advance_projects_endpoint_not_intermediate():
    x = jnp.array([2.0, -2.0, 0.1, 0.4])
    v = jnp.array([4.0, -4.0, 0.2, 1e5])
    t = jnp.array([0.5, 0.5, 0.5, 0.0])
    s = jnp.array([0.8, -0.8, 0.3, 0.7])
    actual = advance(x, v, t, s, 0.5, 0.1)
    expected = jnp.where(t > 0, 0.8 * x + 0.2 * jnp.clip(x + 0.5 * v, -1, 1), s)
    np.testing.assert_allclose(actual, expected, atol=2e-7)
    assert actual[0] > 1 and actual[1] < -1
    assert actual[3] == s[3]


def test_no_projection_when_predicted_endpoint_is_in_range():
    x = jnp.array([0.3, -0.3])
    v = jnp.array([0.2, -0.2])
    t = jnp.ones_like(x)
    np.testing.assert_allclose(advance(x, v, t, x, 0.5, 0.1), x + 0.1 * v, atol=1e-7)


@pytest.mark.parametrize("depth", [1, 6])
def test_original_replay_gaussian_initialization_and_protection(depth):
    s = jnp.ones((1, 128, 128, 1)) * 0.8
    t = jnp.full_like(s, 0.5).at[:, :, :64].set(0)
    key = jax.random.PRNGKey(52)
    expected_initial = initialize(s, t, key, depth)[0]
    first = []

    def predict(images, levels, guides, progress):
        if not first:
            first.extend(images)
        return tuple(jnp.ones_like(x) * 2 for x in images)

    old = np.asarray(sample_grid(predict, s, t, key, depth=depth, steps=8))
    actual, _ = sample(predict, s, t, key, depth=depth, steps=8, projected=False)
    np.testing.assert_array_equal(actual, old)
    first.clear()
    projected, _ = sample(predict, s, t, key, depth=depth, steps=8, projected=True)
    jax.tree.map(np.testing.assert_array_equal, tuple(first), expected_initial)
    assert projected.min() >= -1 - 1e-6 and projected.max() <= 1 + 1e-6
    np.testing.assert_array_equal(projected[:, :, :64], np.asarray(s)[:, :, :64])
    assert max(float(x.max()) for x in first) > 1


def test_one_step_equals_clipped_original_on_active_pixels():
    s = jnp.zeros((2, 32, 32, 1))
    t = jnp.full_like(s, 0.5)
    key = jax.random.PRNGKey(4)

    def predict(images, levels, guides, progress):
        return tuple(4 * jnp.sin(x * 2) for x in images)

    original, _ = sample(predict, s, t, key, depth=2, steps=1, projected=False)
    projected, _ = sample(predict, s, t, key, depth=2, steps=1, projected=True)
    np.testing.assert_allclose(projected, np.clip(original, -1, 1), atol=5e-7)
