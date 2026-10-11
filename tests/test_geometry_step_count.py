import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from geometry_step_count import COUNTS, run_sample
from source_band_fit import run_sample as previous_sample
from test_progress_modulation import modulated


class SourceEndpoint(nnx.Module):
    def __call__(self, x, t, insert, s, u, *, return_network_output=False):
        assert return_network_output
        return (s - x) / jnp.sqrt(jnp.maximum(t, 1e-6))


@pytest.mark.parametrize("steps", COUNTS)
def test_each_step_count_reaches_known_endpoint_and_preserves_mask(steps):
    source = jnp.ones((2, 32, 32, 1)).at[:, 6:26, 13:19].set(-0.8)
    levels = jnp.full_like(source, 0.5).at[:, :, :16].set(0)
    value = run_sample(
        SourceEndpoint(), source, levels, jax.random.PRNGKey(42), 2, f"control_{steps}"
    )
    np.testing.assert_allclose(value, source, atol=2e-7, rtol=0)
    mask = np.asarray(levels) == 0
    np.testing.assert_array_equal(value[mask], np.asarray(source)[mask])
    assert np.isfinite(value).all()


@pytest.mark.parametrize("anchor", ["control", "band"])
def test_real_modulated_model32_replays_frozen_sampler(tmp_path, anchor):
    _, model, _, p = modulated(tmp_path)
    s, t = p["guides"][0], p["levels"][0]
    key = jax.random.PRNGKey(38)
    actual = run_sample(model, s, t, key, 2, f"{anchor}_32")
    expected = previous_sample(model, s, t, key, 2, anchor)
    np.testing.assert_array_equal(actual, expected)
