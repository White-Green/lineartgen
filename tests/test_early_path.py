import jax
import jax.numpy as jnp
import numpy as np
import pytest
from anchored_endpoint import noisy_gradient
from diagnose_learnability import update
from early_path_fit import focus_time, prepared_batch, split_arms
from progress_modulation_fit import prepared_batch as old_batch
from test_progress_modulation import modulated
from velocity_bridge_toy import batch

from lineartgen.diffusion import pyramid
from lineartgen.training import checkpoints, lineart_state
from lineartgen.training.lineart_study import create_arm


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_focus_keeps_inputs_labels_noise_zero_times_and_unselected_values(depth):
    p, _, tau = batch(6001, depth)
    none, none_tau, selected = focus_time(p, tau, jnp.zeros_like(tau, dtype=bool))
    jax.tree.map(np.testing.assert_array_equal, p, none)
    np.testing.assert_array_equal(none_tau, tau)
    assert not np.any(selected)
    q, new, selected = focus_time(p, tau, jnp.ones_like(tau, dtype=bool))
    jax.tree.map(np.testing.assert_array_equal, p["direct"], q["direct"])
    for field in ("target", "levels", "guides"):
        jax.tree.map(np.testing.assert_array_equal, p["bridge"][field], q["bridge"][field])
    np.testing.assert_array_equal(np.asarray(new) == 0, np.asarray(tau) == 0)
    assert np.all(np.asarray(new)[np.asarray(new) > 0] <= 0.05)
    clean = pyramid(p["direct"]["target"], depth)
    for i, (x, z, y, t, s) in enumerate(
        zip(
            q["bridge"]["images"],
            p["direct"]["images"],
            clean,
            p["direct"]["levels"],
            p["direct"]["guides"],
            strict=True,
        )
    ):
        expected = jnp.where(t > 0, (1 - new) * z + new * y, s)
        np.testing.assert_allclose(x, expected, atol=3e-7, rtol=0)
        protected = np.asarray(t) == 0
        np.testing.assert_array_equal(np.asarray(x)[protected], np.asarray(s)[protected])
        zero = np.asarray(tau[:, 0, 0, 0]) == 0
        np.testing.assert_array_equal(
            np.asarray(x)[zero], np.asarray(p["bridge"]["images"][i])[zero]
        )


@pytest.mark.parametrize("step", [6000, 6001, 6002, 11999])
def test_control_is_original_stream_and_early_is_reproducible(step):
    values, row = prepared_batch(step)
    old, old_row = old_batch(step)
    jax.tree.map(np.testing.assert_array_equal, values["control"], old)
    assert row["inputs_sha256"]["control"] == old_row["inputs_sha256"]
    again, again_row = prepared_batch(step)
    jax.tree.map(np.testing.assert_array_equal, values, again)
    assert again_row == row
    assert row["focused_early_count"] >= row["control_early_count"]


def test_actual_optimizer_copy_is_independent_and_restores_next_update(tmp_path):
    _, m, c, p = modulated(tmp_path)
    original = create_arm(c, m)
    original["gradient"] = noisy_gradient
    update(original, p)
    arms = split_arms(original, {name: c for name in ("control", "early")})
    expected = jax.tree.map(
        lambda x: np.asarray(x).copy(), lineart_state.payload({"x": original}, 1)
    )
    for arm in arms.values():
        jax.tree.map(np.testing.assert_array_equal, expected, lineart_state.payload({"x": arm}, 1))
    update(arms["control"], p)
    jax.tree.map(
        np.testing.assert_array_equal, expected, lineart_state.payload({"x": arms["early"]}, 1)
    )
    update(arms["early"], p)
    with checkpoints.manager(tmp_path / "state") as saved:
        lineart_state.save(saved, arms, 2)
        restored = split_arms(original, {name: c for name in ("control", "early")})
        assert lineart_state.restore(saved, restored, 2) == 2
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 2),
        lineart_state.payload(restored, 2),
    )
    for name in arms:
        assert update(arms[name], p) == update(restored[name], p)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 3),
        lineart_state.payload(restored, 3),
    )
