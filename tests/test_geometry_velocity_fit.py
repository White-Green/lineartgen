import shutil

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from diagnose_learnability import update
from geometry_velocity_fit import SEED, make_arms, prepared_batch
from test_velocity_bridge_toy import fixture
from velocity_bridge_toy import initialize, velocity_inputs

from lineartgen.training import checkpoints, lineart_state


@pytest.mark.parametrize("depth", [1, 6])
def test_preparation_supports_full_recursion_downsampled_noise_and_protection(depth):
    clean = jnp.ones((2, 512, 512, 1)).at[:, 100:300, 250:255].set(-1)
    clean = clean.at[:, 100:300, 249].set(0.3)
    source = 1 - 0.3 * (1 - clean)
    strength = jnp.full_like(clean, 0.5).at[:, :, :256].set(0)
    p, tau = prepared_batch(clean, source, strength, 7, depth)
    q, other = prepared_batch(clean, source, strength, 7, depth)
    jax.tree.map(np.testing.assert_array_equal, (p, tau), (q, other))
    assert [x.shape[1] for x in p["direct"]["images"]] == [512 // 2**d for d in range(depth)]
    key = jax.random.fold_in(jax.random.PRNGKey(SEED), 7)
    expected = initialize(source, strength, jax.random.fold_in(key, 2), depth)[0]
    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(a, b, atol=3e-7, rtol=0),
        p["direct"]["images"],
        expected,
    )
    values = velocity_inputs(p)
    np.testing.assert_array_equal(
        values["velocity_target"], p["bridge"]["target"] - p["direct"]["images"][0]
    )
    for x, t, s in zip(
        p["bridge"]["images"], p["bridge"]["levels"], p["bridge"]["guides"], strict=True
    ):
        mask = np.asarray(t) == 0
        np.testing.assert_array_equal(np.asarray(x)[mask], np.asarray(s)[mask])
    np.testing.assert_array_equal(
        np.asarray(values["velocity_target"])[np.asarray(strength) == 0], 0
    )
    assert np.any(np.asarray(clean) == np.float32(0.3))


def test_fresh_arms_real_updates_and_next_update_restore(tmp_path):
    _, prepared, _ = fixture(tmp_path)
    for name in ("mse", "interval"):
        shutil.copyfile(tmp_path / "configs/velocity.json", tmp_path / f"configs/{name}.json")
    arms = make_arms(tmp_path)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload({"a": arms["mse"]}, 0),
        lineart_state.payload({"a": arms["interval"]}, 0),
    )
    before = jax.tree.map(
        lambda a: np.asarray(a).copy(), lineart_state.payload({"a": arms["interval"]}, 0)
    )
    point = update(arms["mse"], prepared)
    jax.tree.map(
        np.testing.assert_array_equal, before, lineart_state.payload({"a": arms["interval"]}, 0)
    )
    interval = update(arms["interval"], prepared)
    assert point["gradient_norm"] > 0 and interval["gradient_norm"] > 0
    assert interval["loss"] <= point["loss"]
    assert "velocity_mse" in point and "interval_velocity_mse" in interval
    with checkpoints.manager(tmp_path / "geometry-checkpoints") as saved:
        lineart_state.save(saved, arms, 1)
        restored = make_arms(tmp_path)
        assert lineart_state.restore(saved, restored) == 1
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 1),
        lineart_state.payload(restored, 1),
    )
    for name in arms:
        assert update(arms[name], prepared) == update(restored[name], prepared)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 2),
        lineart_state.payload(restored, 2),
    )
