import jax
import numpy as np
from diagnose_learnability import update
from local_procedural_rollout import prepared_batch, split_local
from local_recursive_fit import fresh_model, make_arm
from procedural_velocity import prepared_batch as old_prepared
from test_local_recursive import config_pair
from train_procedural_structure import prepare_batch
from velocity_bridge_toy import batch, velocity_inputs

from lineartgen.config import load_config
from lineartgen.training import checkpoints, lineart_state


def test_procedural_source_matches_existing_stream_and_preserves_mask():
    c = load_config(
        "../jax-aux-distance-generalization-20261011/runs/aux-distance-generalization-20261011/configs/control.json"
    )
    preparation = prepare_batch(c)
    for index in (0, 1, 5):
        actual, record = prepared_batch(c, preparation, index, size=64)
        expected, old_record = old_prepared(c, preparation, index, size=64)
        for field in ("images", "levels", "guides", "target"):
            # Old velocity fitting uses a random intermediate input, whereas this
            # rollout starts at u=1. Compare conditions/teachers, not those states.
            if field != "images":
                jax.tree.map(np.testing.assert_array_equal, actual[field], expected[field])
        assert record["clean_sha256"] == old_record["clean_sha256"]
        assert record["teacher_counts"] == old_record["teacher_counts"]
        assert record["data_step"] == old_record["step"]
        for x, t, s in zip(actual["images"], actual["levels"], actual["guides"], strict=True):
            mask = np.asarray(t) == 0
            np.testing.assert_array_equal(np.asarray(x)[mask], np.asarray(s)[mask])
        actual2, record2 = prepared_batch(c, preparation, index, size=64)
        jax.tree.map(np.testing.assert_array_equal, actual, actual2)
        assert record == record2


def test_local_only_continuation_does_not_alias_parent_and_restores_update(tmp_path):
    _, c = config_pair(tmp_path)
    c = c.model_copy(
        update={
            "train": c.train.model_copy(
                update={
                    "steps": 6000,
                    "learning_rate_decay_steps": 1200,
                    "warmup_steps": 0,
                    "batch_size": 4,
                    "accumulate_steps": 1,
                    "learning_rate": 5e-5,
                    "minimum_learning_rate": 1e-4,
                }
            )
        }
    )
    parent = make_arm(c, fresh_model(c))
    p = velocity_inputs(batch(2, 2, count=2)[0])
    assert update(parent, p)["gradient_norm"] > 0
    extended = c.model_copy(update={"train": c.train.model_copy(update={"steps": 10000})})
    arm = split_local(parent, extended)
    saved_parent = jax.tree.map(
        lambda x: np.asarray(x).copy(), lineart_state.payload({"local": parent}, 1)
    )
    assert update(arm, p)["gradient_norm"] > 0
    jax.tree.map(
        np.testing.assert_array_equal, saved_parent, lineart_state.payload({"local": parent}, 1)
    )
    update(parent, p)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload({"local": parent}, 2),
        lineart_state.payload({"local": arm}, 2),
    )
    with checkpoints.manager(tmp_path / "checkpoints") as saved:
        lineart_state.save(saved, {"local": arm}, 2)
        restored = split_local(parent, extended)
        assert lineart_state.restore(saved, {"local": restored}, 2) == 2
    assert update(arm, p) == update(restored, p)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload({"local": arm}, 3),
        lineart_state.payload({"local": restored}, 3),
    )
    for n in (0, 6000, 9999):
        np.testing.assert_array_equal(arm["schedule"](n), parent["schedule"](n))
