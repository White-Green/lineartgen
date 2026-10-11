import shutil

import jax
import numpy as np
from diagnose_learnability import update
from flax import nnx
from fresh_geometry_fit import make_arms
from test_velocity_bridge_toy import fixture

from lineartgen.training import checkpoints, lineart_state


def test_fresh_arm_updates_and_full_state_restore(tmp_path):
    initial, prepared, _ = fixture(tmp_path)
    shutil.copyfile(tmp_path / "configs/velocity.json", tmp_path / "configs/fresh.json")
    arms = make_arms(tmp_path)
    assert set(arms) == {"fresh"}
    arm = arms["fresh"]
    jax.tree.map(
        np.testing.assert_array_equal,
        nnx.state(initial, nnx.Param),
        nnx.state(arm["model"], nnx.Param),
    )
    assert int(arm["optimizer"].step[...]) == 0
    jax.tree.map(np.testing.assert_array_equal, arm["ema"], nnx.state(initial, nnx.Param))
    metrics = update(arm, prepared)
    assert metrics["gradient_norm"] > 0 and "interval_velocity_mse" in metrics
    with checkpoints.manager(tmp_path / "fresh-checkpoints") as manager:
        lineart_state.save(manager, arms, 1)
        restored = make_arms(tmp_path)
        assert lineart_state.restore(manager, restored) == 1
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 1),
        lineart_state.payload(restored, 1),
    )
    assert update(arms["fresh"], prepared) == update(restored["fresh"], prepared)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 2),
        lineart_state.payload(restored, 2),
    )
