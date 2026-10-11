import jax
import jax.numpy as jnp
import numpy as np
from diagnose_learnability import update
from flax import nnx
from geometry_velocity_fit import prepared_batch
from interval_velocity_fit import interval_gradient
from start_focused_fit import METHODS, SEED, focus_start, split_arms
from test_velocity_bridge_toy import fixture

from lineartgen.config import load_config
from lineartgen.training import checkpoints, lineart_state
from lineartgen.training.lineart_study import create_arm


def test_changed_atom_preserves_direct_and_unchanged_bridge_exactly():
    clean = jnp.ones((16, 64, 64, 1)).at[:, 8:56, 30:33].set(-1)
    source = 1 - 0.3 * (1 - clean)
    strength = jnp.full_like(clean, 0.5).at[:, :, :32].set(0)
    counts = [0, 0, 0]
    for step in range(6000, 6032):
        batch, tau = prepared_batch(clean, source, strength, step, 3)
        focused, other = focus_start(batch, tau, step)
        jax.tree.map(np.testing.assert_array_equal, focused["direct"], batch["direct"])
        key = jax.random.fold_in(jax.random.PRNGKey(SEED), step)
        draws = jax.random.uniform(jax.random.fold_in(key, 1), (16, 2, 1, 1, 1))
        selected = np.asarray(draws[:, 0, 0, 0, 0] < 0.75)
        for a, b, c in zip(
            jax.tree.leaves(focused["bridge"]),
            jax.tree.leaves(batch["bridge"]),
            jax.tree.leaves(batch["direct"]),
            strict=True,
        ):
            np.testing.assert_array_equal(np.asarray(a)[~selected], np.asarray(b)[~selected])
            np.testing.assert_array_equal(np.asarray(a)[selected], np.asarray(c)[selected])
        assert np.all(np.asarray(other)[selected] == 0)
        np.testing.assert_array_equal(np.asarray(other)[~selected], np.asarray(tau)[~selected])
        counts[0] += int((tau == 0).sum())
        counts[1] += int((other == 0).sum())
        counts[2] += len(tau)
    assert 0.18 < counts[0] / counts[2] < 0.32
    assert 0.68 < counts[1] / counts[2] < 0.82


def test_continue_real_optimizer_ema_with_common_new_schedule(tmp_path):
    model, prepared, _ = fixture(tmp_path)
    config = load_config(tmp_path / "configs/velocity.json")
    reference = create_arm(config, model)
    reference["gradient"] = interval_gradient
    update(reference, prepared)
    configs = {
        name: config.model_copy(
            update={
                "train": config.train.model_copy(
                    update={
                        "learning_rate": 5e-6,
                        "batch_size": 4,
                        "minimum_learning_rate": 1e-5,
                        "warmup_steps": 0,
                        "learning_rate_decay_steps": 6000,
                    }
                )
            }
        )
        for name in METHODS
    }
    arms = split_arms(reference, configs)
    for arm in arms.values():
        jax.tree.map(
            np.testing.assert_array_equal,
            lineart_state.payload({"a": reference}, 1),
            lineart_state.payload({"a": arm}, 1),
        )
        assert abs(float(arm["schedule"](6000)) - 1e-5) < 1e-11
    a = update(arms["control"], prepared)
    b = update(arms["focused"], prepared)
    assert a == b and a["gradient_norm"] > 0
    assert int(arms["control"]["optimizer"].step[...]) == 2
    with checkpoints.manager(tmp_path / "continue-checkpoint") as manager:
        lineart_state.save(manager, arms, 2)
        restored = split_arms(reference, configs)
        assert lineart_state.restore(manager, restored) == 2
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 2),
        lineart_state.payload(restored, 2),
    )
    for name in METHODS:
        assert update(arms[name], prepared) == update(restored[name], prepared)
        jax.tree.map(
            np.testing.assert_array_equal,
            nnx.state(arms[name]["model"]),
            nnx.state(restored[name]["model"]),
        )
