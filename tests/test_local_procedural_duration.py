import jax
import numpy as np
from diagnose_learnability import update
from flax import nnx
from local_procedural_duration import prepared_batch, split_local
from local_procedural_rollout import prepared_batch as prior_prepared
from local_recursive_fit import fresh_model, make_arm
from test_local_recursive import config_pair
from train_procedural_structure import prepare_batch
from velocity_bridge_toy import batch, velocity_inputs

from lineartgen.config import load_config
from lineartgen.training import checkpoints, lineart_state


def test_continued_data_indices_keep_the_existing_stream():
    config = load_config(
        "../jax-aux-distance-generalization-20261011/runs/aux-distance-generalization-20261011/configs/control.json"
    )
    preparation = prepare_batch(config)
    for index in (4000, 4001, 13999):
        p, row = prepared_batch(config, preparation, index, size=64)
        old, old_row = prior_prepared(config, preparation, index, size=64)
        jax.tree.map(np.testing.assert_array_equal, p, old)
        assert row == old_row and row["data_step"] == index + 1
        assert len(p["images"]) == index % 6 + 1


def test_schedule_change_preserves_state_but_changes_update_and_survives_restore(tmp_path):
    _, c = config_pair(tmp_path)
    c = c.model_copy(
        update={
            "train": c.train.model_copy(
                update={
                    "steps": 10000,
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
    update(parent, p)
    continued = c.model_copy(
        update={
            "train": c.train.model_copy(
                update={
                    "steps": 20000,
                    "learning_rate_decay_steps": 20000,
                    "minimum_learning_rate": 1e-5,
                }
            )
        }
    )
    production_arm = split_local(parent, continued)
    np.testing.assert_allclose(production_arm["schedule"](10000), 5.5e-5, rtol=1e-6)
    np.testing.assert_allclose(production_arm["schedule"](20000), 1e-5, rtol=1e-6)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload({"local": parent}, 1),
        lineart_state.payload({"local": production_arm}, 1),
    )
    # Shorten only the test schedule to measure its effect at the real fixture
    # optimizer count=1 without fabricating moments or doing 10,000 test updates.
    fixture = continued.model_copy(
        update={
            "train": continued.train.model_copy(
                update={
                    "learning_rate_decay_steps": 4,
                }
            )
        }
    )
    arm = split_local(parent, fixture)
    before = jax.tree.map(lambda x: np.asarray(x).copy(), nnx.state(parent["model"], nnx.Param))
    assert update(parent, p)["loss"] == update(arm, p)["loss"]
    old_delta = jax.tree.map(
        lambda x, y: np.asarray(x) - y, nnx.state(parent["model"], nnx.Param), before
    )
    new_delta = jax.tree.map(
        lambda x, y: np.asarray(x) - y, nnx.state(arm["model"], nnx.Param), before
    )
    def norm(tree):
        return sum(float(np.square(x).sum()) for x in jax.tree.leaves(tree)) ** 0.5

    assert norm(new_delta) < 0.95 * norm(old_delta)
    with checkpoints.manager(tmp_path / "checkpoints") as saved:
        lineart_state.save(saved, {"local": arm}, 2)
        restored = split_local(parent, fixture)
        assert lineart_state.restore(saved, {"local": restored}, 2) == 2
    assert update(arm, p) == update(restored, p)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload({"local": arm}, 3),
        lineart_state.payload({"local": restored}, 3),
    )
