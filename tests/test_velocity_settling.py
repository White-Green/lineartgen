import jax
import numpy as np
from diagnose_learnability import update
from flax import nnx
from interval_velocity_fit import gradient
from test_velocity_bridge_toy import fixture
from velocity_settling import METHODS, split_arms

from lineartgen.config import load_config
from lineartgen.training import checkpoints, lineart_state
from lineartgen.training.lineart_study import create_arm


def test_identical_full_state_then_distinct_rates_and_exact_restore(tmp_path):
    model, prepared, _ = fixture(tmp_path)
    c = load_config(tmp_path / "configs/velocity.json")
    reference = create_arm(c, model)
    reference["gradient"] = gradient
    update(reference, prepared)
    configs = {
        name: c.model_copy(
            update={
                "train": c.train.model_copy(
                    update={
                        "batch_size": 4,
                        "learning_rate": 5e-5 if name == "control" else 5e-6,
                        "minimum_learning_rate": 1e-4 if name == "control" else 1e-5,
                        "warmup_steps": 0,
                        "learning_rate_decay_steps": 6000,
                    }
                )
            }
        )
        for name in METHODS
    }
    arms = split_arms(reference, configs)
    for name, arm in arms.items():
        jax.tree.map(
            np.testing.assert_array_equal,
            lineart_state.payload({"a": reference}, 1),
            lineart_state.payload({"a": arm}, 1),
        )
        np.testing.assert_allclose(
            arm["schedule"](6000), 1e-4 if name == "control" else 1e-5, rtol=1e-6
        )
    before = jax.tree.map(
        lambda x: np.asarray(x).copy(), lineart_state.payload({"a": arms["settled"]}, 1)
    )
    a = update(arms["control"], prepared)
    jax.tree.map(
        np.testing.assert_array_equal, before, lineart_state.payload({"a": arms["settled"]}, 1)
    )
    b = update(arms["settled"], prepared)
    assert a == b and a["gradient_norm"] > 0
    assert any(
        np.any(a != b)
        for a, b in zip(
            jax.tree.leaves(nnx.state(arms["control"]["model"])),
            jax.tree.leaves(nnx.state(arms["settled"]["model"])),
            strict=True,
        )
    )
    with checkpoints.manager(tmp_path / "checkpoints") as saved:
        lineart_state.save(saved, arms, 2)
        restored = split_arms(reference, configs)
        assert lineart_state.restore(saved, restored) == 2
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 2),
        lineart_state.payload(restored, 2),
    )
    for name in METHODS:
        assert update(arms[name], prepared) == update(restored[name], prepared)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 3),
        lineart_state.payload(restored, 3),
    )
