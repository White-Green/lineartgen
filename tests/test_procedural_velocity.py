import jax
import numpy as np
import pytest
from aux_distance_generalization import prepared_batch as source_batch
from diagnose_learnability import snapshot, update
from flax import nnx
from geometry_velocity_fit import prepared_batch as fixed_bridge_batch
from procedural_velocity import make_arms, prepared_batch, teachers
from test_velocity_bridge_toy import fixture
from train_procedural_structure import prepare_batch
from velocity_bridge_toy import velocity_inputs

from lineartgen.config import load_config, save_config
from lineartgen.models.serialization import load_inference
from lineartgen.training import checkpoints, lineart_state


def source_config(tmp_path):
    fixture(tmp_path)
    c = load_config(tmp_path / "configs/velocity.json")
    return c.model_copy(
        update={
            "geometry": c.geometry.model_copy(
                update=dict(probability=0.9, coverage_probability=1.0)
            ),
            "sketch": c.sketch.model_copy(
                update=dict(mode="mixed", probability=0.8, strong_probability=0.5)
            ),
            "matching": c.matching.model_copy(update={"method": "independent"}),
            "diffusion": c.diffusion.model_copy(update={"training_path": "rough"}),
        }
    )


@pytest.mark.parametrize("step", [0, 5])
def test_procedural_targets_sources_and_masks_match_BL_and_bridge_matches_BC(tmp_path, step):
    c = source_config(tmp_path)
    prepare = prepare_batch(c)
    old, row = source_batch(c, prepare, step, 512)
    actual, actual_row = prepared_batch(c, prepare, step, 512)
    assert row == actual_row and actual_row["depth"] == step + 1
    for k in ("target", "levels", "guides"):
        jax.tree.map(np.testing.assert_array_equal, old[k], actual[k])
    expected = velocity_inputs(
        fixed_bridge_batch(old["target"], old["guides"][0], old["levels"][0], step, step + 1)[0]
    )
    jax.tree.map(np.testing.assert_array_equal, actual, expected)
    for x, t, s in zip(actual["images"], actual["levels"], actual["guides"], strict=True):
        mask = np.asarray(t) == 0
        np.testing.assert_array_equal(np.asarray(x)[mask], np.asarray(s)[mask])
    mask = np.asarray(actual["levels"][0]) == 0
    np.testing.assert_array_equal(np.asarray(actual["velocity_target"])[mask], 0)
    clean, _ = teachers(c, step, 512)
    np.testing.assert_array_equal(np.asarray(actual["target"])[~mask], clean[~mask])


def test_both_real_updates_restore_and_export_full_velocity_model(tmp_path):
    c = source_config(tmp_path)
    for name in ("mse", "interval"):
        save_config(c, tmp_path / f"configs/{name}.json")
    p, _ = prepared_batch(c, prepare_batch(c), 1, 64)
    arms = make_arms(tmp_path)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload({"a": arms["mse"]}, 0),
        lineart_state.payload({"a": arms["interval"]}, 0),
    )
    for arm in arms.values():
        m = update(arm, p)
        assert m["gradient_norm"] > 0 and np.isfinite(m["loss"])
    with checkpoints.manager(tmp_path / "checkpoints") as saved:
        lineart_state.save(saved, arms, 1)
        restored = make_arms(tmp_path)
        assert lineart_state.restore(saved, restored, 1) == 1
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 1),
        lineart_state.payload(restored, 1),
    )
    snapshot(tmp_path, arms, 1)
    for name, arm in arms.items():
        for mode in ("raw", "ema"):
            model, config = load_inference(tmp_path / f"snapshots/step-1/{name}/{mode}")
            assert config == c
            jax.tree.map(
                np.testing.assert_array_equal,
                nnx.state(model, nnx.Param),
                arm["ema"] if mode == "ema" else nnx.state(arm["model"], nnx.Param),
            )
        assert update(arm, p) == update(restored[name], p)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 2),
        lineart_state.payload(restored, 2),
    )
