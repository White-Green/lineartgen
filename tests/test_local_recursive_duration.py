import jax
import numpy as np
from diagnose_learnability import update
from local_recursive_duration import split_arms
from local_recursive_fit import fresh_model, make_arm
from test_local_recursive import config_pair
from velocity_bridge_toy import batch, velocity_inputs

from lineartgen.training import checkpoints, lineart_state


def test_each_architecture_inherits_full_state_and_restores_next_update(tmp_path):
    configs = dict(zip(("unet", "local"), config_pair(tmp_path), strict=True))
    for name, c in configs.items():
        configs[name] = c.model_copy(
            update={
                "train": c.train.model_copy(
                    update={
                        "steps": 1200,
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
    parents = {n: make_arm(c, fresh_model(c)) for n, c in configs.items()}
    p = velocity_inputs(batch(2, 2, count=2)[0])
    for a in parents.values():
        assert update(a, p)["gradient_norm"] > 0
    extended = {
        n: c.model_copy(update={"train": c.train.model_copy(update={"steps": 6000})})
        for n, c in configs.items()
    }
    arms = split_arms(parents, extended)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(parents, 1),
        lineart_state.payload(arms, 1),
    )
    before = jax.tree.map(lambda x: np.asarray(x).copy(), lineart_state.payload(parents, 1))
    for n in parents:
        for step in (0, 1, 1200, 5999):
            np.testing.assert_array_equal(arms[n]["schedule"](step), parents[n]["schedule"](step))
        update(arms[n], p)
    jax.tree.map(np.testing.assert_array_equal, before, lineart_state.payload(parents, 1))
    for n in parents:
        update(parents[n], p)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(parents, 2),
        lineart_state.payload(arms, 2),
    )
    with checkpoints.manager(tmp_path / "continued") as saved:
        lineart_state.save(saved, arms, 2)
        restored = split_arms(parents, extended)
        assert lineart_state.restore(saved, restored, 2) == 2
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 2),
        lineart_state.payload(restored, 2),
    )
    for n in arms:
        assert update(arms[n], p) == update(restored[n], p)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 3),
        lineart_state.payload(restored, 3),
    )
