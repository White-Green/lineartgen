import jax
import jax.numpy as jnp
import numpy as np
import pytest
from anchored_endpoint import velocity as capped_velocity
from bridge_time_grid import sample_grid, time_nodes
from endpoint_time_grid import velocity
from test_anchored_endpoint import ZeroHead
from test_velocity_bridge_toy import fixture
from velocity_bridge_toy import sample


def test_small_remaining_time_uses_actual_denominator_and_preserves_mask():
    x = jnp.full((1, 16, 16, 1), 0.4)
    s = jnp.full_like(x, 0.8)
    t = jnp.full_like(x, 0.5).at[:, :, :8].set(0)
    u = jnp.full_like(x, 0.005)
    args = (ZeroHead(), (x,), (t,), (s,), (u,), True)
    actual = velocity(*args)[0]
    old = capped_velocity(*args)[0]
    np.testing.assert_allclose(actual[:, :, 8:], 80, atol=1e-5, rtol=0)
    np.testing.assert_allclose(old[:, :, 8:], 12.8, atol=1e-6, rtol=0)
    np.testing.assert_array_equal(actual[:, :, :8], 0)


@pytest.mark.parametrize("steps", [32, 128])
def test_source_endpoint_converges_with_sine_grid_and_exact_protection(steps):
    assert np.min(1 - time_nodes(steps)[:-1]) > np.finfo(np.float32).eps
    s = jnp.ones((2, 32, 32, 1)).at[:, 5:25, 14:18].set(0.4)
    t = jnp.full_like(s, 0.5).at[:, :, :16].set(0)
    result = sample_grid(
        lambda x, t, s, u: velocity(ZeroHead(), x, t, s, u, True),
        s,
        t,
        jax.random.PRNGKey(8),
        depth=2,
        steps=steps,
    )
    np.testing.assert_allclose(result, s, atol=2e-7, rtol=0)
    np.testing.assert_array_equal(np.asarray(result)[:, :, :16], np.asarray(s)[:, :, :16])


@pytest.mark.parametrize("source_anchor", [False, True])
def test_real_model_uniform32_is_bitwise_unchanged(tmp_path, source_anchor):
    model, p, _ = fixture(tmp_path)
    s, t = p["guides"][0], p["levels"][0]
    key = jax.random.PRNGKey(19)
    results = [
        sample(
            lambda x, t, s, u, fn=fn: fn(model, x, t, s, u, source_anchor),
            s,
            t,
            key,
            depth=2,
            steps=32,
        )
        for fn in (capped_velocity, velocity)
    ]
    np.testing.assert_array_equal(*results)
