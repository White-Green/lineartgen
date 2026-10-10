import shutil

import jax
import numpy as np
from bridge_duration import METHODS, batch_for, make_arms
from campaign_protocol import write
from diagnose_learnability import update
from flax import nnx
from test_velocity_bridge_toy import fixture
from velocity_bridge_toy import make_arms as original_arms
from velocity_bridge_toy import velocity_inputs

from lineartgen.config import save_config
from lineartgen.models.serialization import save_inference
from lineartgen.training import checkpoints, lineart_state


def test_two_parent_full_states_schedules_and_next_updates_are_retained(tmp_path):
    root = tmp_path / "duration"
    (root / "configs").mkdir(parents=True)
    parents = {}
    controls = {}
    for name in METHODS:
        method = name.removesuffix("_long")
        p = tmp_path / method
        p.mkdir()
        fixture(p)
        arm = original_arms(p)["velocity"]
        for n in range(2):
            update(arm, velocity_inputs(batch_for(name, n, 2, count=2)[0]))
        with checkpoints.manager(p / "training/checkpoints") as saved:
            lineart_state.save(saved, {method: arm}, 2)
        config = arm["config"]
        for mode in ("raw", "ema"):
            model = nnx.clone(arm["model"])
            if mode == "ema":
                nnx.update(model, arm["ema"])
            save_inference(
                model, config, p / f"snapshots/step-2/{method}/{mode}", provenance="fixture"
            )
        new = config.model_copy(
            update={
                "train": config.train.model_copy(
                    update={"steps": 30000, "learning_rate_decay_steps": config.train.steps}
                )
            }
        )
        save_config(new, root / f"configs/{name}.json")
        shutil.copytree(p / f"snapshots/step-2/{method}/raw", root / "initial" / name)
        parents[name] = dict(root=str(p), method=method)
        controls[name] = arm
    write(root / "plan.json", dict(parents=parents, start_step=2))
    arms = make_arms(root)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 2),
        lineart_state.payload(controls, 2),
    )
    for name in METHODS:
        for n in (0, 100, 5999, 6000, 12000, 30000):
            assert float(arms[name]["schedule"](n)) == float(controls[name]["schedule"](n))
        p = velocity_inputs(batch_for(name, 2, 2, count=2)[0])
        assert update(arms[name], p) == update(controls[name], p)
        assert int(arms[name]["optimizer"].step[...]) == 3
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 3),
        lineart_state.payload(controls, 3),
    )
    with checkpoints.manager(root / "training/checkpoints") as saved:
        lineart_state.save(saved, arms, 3)
        restored = make_arms(root)
        assert lineart_state.restore(saved, restored) == 3
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 3),
        lineart_state.payload(restored, 3),
    )
    for name in METHODS:
        p = velocity_inputs(batch_for(name, 3, 2, count=2)[0])
        assert update(arms[name], p) == update(restored[name], p)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 4),
        lineart_state.payload(restored, 4),
    )
