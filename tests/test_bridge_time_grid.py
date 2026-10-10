import jax
import jax.numpy as jnp
import numpy as np
import pytest
from bridge_time_grid import sample_grid, time_nodes
from velocity_bridge_toy import initialize, sample


@pytest.mark.parametrize("steps", [1, 8, 32, 256])
def test_nodes_and_constant_velocity_integrate_to_same_endpoint(steps):
    nodes = time_nodes(steps)
    assert nodes[0] == 0 and nodes[-1] == 1 and np.all(np.diff(nodes) > 0)
    if steps > 1:
        assert nodes[1] < 1 / steps and 1 - nodes[-2] < 1 / steps
    source = jnp.ones((2, 32, 32, 1))
    strength = jnp.full_like(source, 0.5).at[:, :, :16].set(0)
    key = jax.random.PRNGKey(7)
    start = initialize(source, strength, key, 2)[0][0]
    times = []

    def predict(images, levels, guides, progress):
        times.append(float(progress[0][0, 0, 0, 0]))
        return tuple(jnp.full_like(x, 0.25) for x in images)

    result = sample_grid(predict, source, strength, key, depth=2, steps=steps)
    np.testing.assert_allclose(
        result, jnp.where(strength > 0, start + 0.25, source), atol=2e-5, rtol=0
    )
    np.testing.assert_array_equal(np.asarray(result)[:, :, :16], 1)
    np.testing.assert_allclose(times, 1 - nodes[:-1], atol=3e-8, rtol=0)


def test_single_step_matches_original_solver_exactly():
    source = jnp.ones((2, 32, 32, 1)) * 0.3
    strength = jnp.full_like(source, 0.1).at[:, :, :16].set(0)
    key = jax.random.PRNGKey(17)

    def predict(x, t, s, u):
        return tuple(0.7 * v + 0.1 * w for v, w in zip(x, s, strict=True))

    a = sample(predict, source, strength, key, depth=2, steps=1)
    b = sample_grid(predict, source, strength, key, depth=2, steps=1)
    np.testing.assert_array_equal(a, b)


def test_invalid_grid_count_rejected():
    with pytest.raises(ValueError):
        time_nodes(0)
