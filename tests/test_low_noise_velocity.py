import shutil

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from bridge_time_grid import sample_grid as original_sample
from diagnose_learnability import update
from geometry_velocity_fit import prepared_batch as original_batch
from low_noise_velocity_fit import make_arms
from low_noise_velocity_path import initialize, path_inputs, prepared_batch, sample_grid
from test_velocity_bridge_toy import fixture
from velocity_bridge_toy import velocity_inputs

from lineartgen.diffusion import pyramid
from lineartgen.training import checkpoints, lineart_state


@pytest.mark.parametrize("depth", [1, 6])
def test_original_preparation_exact_and_new_noise_scaled_after_resize(depth):
    y = jnp.ones((2, 512, 512, 1)).at[:, 100:300, 275:278].set(-0.5)
    s = 1 - 0.3 * (1 - y)
    t = jnp.full_like(y, 0.5).at[:, :, :256].set(0)
    p, tau = original_batch(y, s, t, 1, depth)
    q, qtau = prepared_batch(y, s, t, 1, depth, noise_scale=0.25)
    jax.tree.map(np.testing.assert_array_equal, (p, tau), (q, qtau))
    low, ltau = prepared_batch(y, s, t, 1, depth)
    np.testing.assert_array_equal(tau, ltau)
    for name in ("direct", "bridge"):
        for field in ("guides", "levels", "progress", "target"):
            jax.tree.map(np.testing.assert_array_equal, p[name][field], low[name][field])
    for a, b, guide in zip(
        p["direct"]["images"], low["direct"]["images"], low["direct"]["guides"], strict=True
    ):
        np.testing.assert_allclose(b, guide + 0.2 * (a - guide), atol=3e-7, rtol=0)
    if depth > 1:
        assert not np.allclose(
            low["direct"]["images"][1], pyramid(low["direct"]["images"][0], 2)[1]
        )
    key = jax.random.PRNGKey(5)
    a = path_inputs(s, y, t, key, jnp.full((2, 1, 1, 1), 0.2), depth, 0.05)
    b = path_inputs(s, y, t, key, jnp.full((2, 1, 1, 1), 0.3), depth, 0.05)
    z = initialize(s, t, key, depth, 0.05)[0]
    target = pyramid(jnp.where(t > 0, y, s), depth)
    for aa, bb, zz, yy in zip(a["images"], b["images"], z, target, strict=True):
        np.testing.assert_allclose((bb - aa) / 0.1, yy - zz, atol=2e-6, rtol=0)


@pytest.mark.parametrize("amplitude", [0.25, 0.05])
def test_same_grid_oracle_and_protected_source(amplitude):
    s = jnp.ones((2, 32, 32, 1)) * 0.7
    t = jnp.full_like(s, 0.5).at[:, :, :16].set(0)
    y = jnp.full_like(s, -0.5)
    key = jax.random.PRNGKey(20)

    def predictor(images, levels, guides, progress):
        return tuple(
            (jnp.full_like(x, -0.5) - x) / u for x, u in zip(images, progress, strict=True)
        )

    a = sample_grid(predictor, s, t, key, depth=2, steps=32, noise_scale=amplitude)
    np.testing.assert_allclose(a, jnp.where(t > 0, y, s), atol=1e-6, rtol=0)
    np.testing.assert_array_equal(
        np.asarray(a)[np.asarray(t) == 0], np.asarray(s)[np.asarray(t) == 0]
    )
    if amplitude == 0.25:
        np.testing.assert_array_equal(a, original_sample(predictor, s, t, key, depth=2, steps=32))


def test_actual_low_noise_update_and_restore(tmp_path):
    _, _, direct = fixture(tmp_path)
    shutil.copyfile(tmp_path / "configs/velocity.json", tmp_path / "configs/low_noise.json")
    p = velocity_inputs(
        prepared_batch(direct["target"], direct["guides"][0], direct["levels"][0], 1, 2)[0]
    )
    arms = make_arms(tmp_path)
    m = update(arms["low_noise"], p)
    assert m["gradient_norm"] > 0 and m["velocity_mse"] > 0
    with checkpoints.manager(tmp_path / "checkpoints") as saved:
        lineart_state.save(saved, arms, 1)
        restored = make_arms(tmp_path)
        assert lineart_state.restore(saved, restored) == 1
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 1),
        lineart_state.payload(restored, 1),
    )
    assert update(arms["low_noise"], p) == update(restored["low_noise"], p)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 2),
        lineart_state.payload(restored, 2),
    )
