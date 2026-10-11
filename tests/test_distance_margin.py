import shutil

import jax
import jax.numpy as jnp
import numpy as np
from diagnose_learnability import update
from distance_margin_fit import make_arms
from test_distance_supervision import fixture, sample_target

from lineartgen.training import checkpoints, lineart_state
from lineartgen.training.distance_margin import distance_margin_supervision
from lineartgen.training.distance_supervision import distance_supervision


def test_every_saturated_pixel_allows_overshoot_including_near_boundary():
    clean, d = sample_target()
    p = jnp.where(clean == 1, 2 * d + 2, jnp.where(clean == -1, 2 * d - 2, 2 * d))
    active = jnp.ones_like(clean, bool)
    value, g = jax.value_and_grad(distance_margin_supervision)(p, d, clean, active)
    assert value == 0
    np.testing.assert_array_equal(g, 0)
    assert distance_supervision(p, d, clean, active) > 0


def test_wrong_side_descent_and_fractional_target_remain_live():
    clean, d = sample_target()
    p = -2 * d
    active = jnp.ones_like(clean, bool)
    value, g = jax.value_and_grad(distance_margin_supervision)(p, d, clean, active)
    assert value > 0 and np.isfinite(g).all()
    assert distance_margin_supervision(p - g, d, clean, active) < value
    aa = (clean > -1) & (clean < 1)
    p = 2 * d + jnp.where(aa, 0.5, 0)
    value, g = jax.value_and_grad(distance_margin_supervision)(p, d, clean, active)
    assert value > 0 and np.all(np.asarray(g)[np.asarray(aa)] > 0)


def test_protected_gradients_zero():
    clean, d = sample_target()
    active = jnp.ones_like(clean, bool).at[:, :, :16].set(False)
    _, g = jax.value_and_grad(distance_margin_supervision)(-2 * d, d, clean, active)
    np.testing.assert_array_equal(g[:, :, :16], 0)
    value, g = jax.value_and_grad(distance_margin_supervision)(
        -2 * d, d, clean, jnp.zeros_like(active)
    )
    assert value == 0
    np.testing.assert_array_equal(g, 0)


def test_real_margin_update_and_restore(tmp_path):
    _, _, p = fixture(tmp_path)
    p["margin"] = p.pop("distance")
    shutil.copyfile(tmp_path / "configs/distance.json", tmp_path / "configs/margin.json")
    arms = make_arms(tmp_path)
    value = update(arms["margin"], p)
    assert value["gradient_norm"] > 0 and value["distance_margin"] > 0
    with checkpoints.manager(tmp_path / "margin-checkpoints") as saved:
        lineart_state.save(saved, arms, 1)
        restored = make_arms(tmp_path)
        assert lineart_state.restore(saved, restored) == 1
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 1),
        lineart_state.payload(restored, 1),
    )
    assert update(arms["margin"], p) == update(restored["margin"], p)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 2),
        lineart_state.payload(restored, 2),
    )
