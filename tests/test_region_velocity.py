import shutil

import jax
import jax.numpy as jnp
import numpy as np
from diagnose_learnability import update
from region_velocity_fit import make_arms
from test_velocity_bridge_toy import fixture

from lineartgen.training import checkpoints, lineart_state
from lineartgen.training.region_velocity import region_masks, region_velocity_mse


def example():
    clean = jnp.asarray([1, 1, 1, 1, 1, 1, 0.2, -0.3, -1], jnp.float32).reshape(1, 3, 3, 1)
    return clean, jnp.ones_like(clean, bool)


def test_equal_region_importance_despite_different_pixel_counts():
    clean, active = example()
    prediction = jnp.where(clean == 1, 1, jnp.where(clean == -1, 3, 2))
    value = region_velocity_mse(prediction, jnp.zeros_like(clean), clean, active)
    np.testing.assert_allclose(value, (1 + 4 + 9) / 3)
    value, g = jax.value_and_grad(region_velocity_mse)(
        jnp.ones_like(clean), jnp.zeros_like(clean), clean, active
    )
    np.testing.assert_allclose(value, 1)
    for mask in region_masks(clean, active):
        np.testing.assert_allclose((jnp.abs(g) * mask).sum(), 2 / 3)


def test_missing_regions_and_all_protected_are_finite_and_correct():
    for y in (-1.0, -0.5, 1.0):
        clean = jnp.full((2, 4, 4, 1), y)
        active = jnp.ones_like(clean, bool)
        p = jnp.full_like(clean, 2)
        value, g = jax.value_and_grad(region_velocity_mse)(p, jnp.zeros_like(p), clean, active)
        assert value == 4 and np.isfinite(g).all()
        value, g = jax.value_and_grad(region_velocity_mse)(p, jnp.zeros_like(p), clean, ~active)
        assert value == 0
        np.testing.assert_array_equal(g, 0)


def test_exact_velocity_target_and_both_residual_signs_have_correct_gradients():
    clean, active = example()
    active = active.at[:, 0].set(False)
    target = clean * 2 + 0.3
    for offset in (-2.0, 0.0, 2.0):
        p = target + offset
        value, g = jax.value_and_grad(region_velocity_mse)(p, target, clean, active)
        np.testing.assert_array_equal(g[:, 0], 0)
        if offset == 0:
            assert value == 0
            np.testing.assert_array_equal(g, 0)
        else:
            assert np.all(np.sign(np.asarray(g)[np.asarray(active)]) == np.sign(offset))
            assert region_velocity_mse(p - g, target, clean, active) < value


def test_actual_region_update_and_full_restore(tmp_path):
    _, prepared, _ = fixture(tmp_path)
    shutil.copyfile(tmp_path / "configs/velocity.json", tmp_path / "configs/balanced.json")
    arms = make_arms(tmp_path)
    metrics = update(arms["balanced"], prepared)
    assert metrics["gradient_norm"] > 0 and metrics["region_velocity_mse"] > 0
    with checkpoints.manager(tmp_path / "region-checkpoints") as saved:
        lineart_state.save(saved, arms, 1)
        restored = make_arms(tmp_path)
        assert lineart_state.restore(saved, restored) == 1
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 1),
        lineart_state.payload(restored, 1),
    )
    assert update(arms["balanced"], prepared) == update(restored["balanced"], prepared)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 2),
        lineart_state.payload(restored, 2),
    )
