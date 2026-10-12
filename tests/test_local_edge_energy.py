import jax
import jax.numpy as jnp
import numpy as np
from anchored_endpoint import endpoints
from diagnose_learnability import update
from flax import nnx
from local_edge_energy import DATA_START, STEPS, prepared_batch, sibling_configs, split_local
from local_procedural_duration import prepared_batch as previous_batch
from local_recursive_fit import fresh_model, make_arm
from rollout_endpoint import midpoint
from rollout_image_loss import objective
from test_local_recursive import config_pair
from train_procedural_structure import prepare_batch
from velocity_bridge_toy import batch, velocity_inputs

from lineartgen.config import ExperimentConfig, load_config
from lineartgen.training import checkpoints, lineart_state
from lineartgen.training.edge_energy import edge_energy_loss


def fixture(tmp_path):
    _, c = config_pair(tmp_path)
    c = c.model_copy(
        update={
            "train": c.train.model_copy(
                update={
                    "steps": 20000,
                    "learning_rate_decay_steps": 20000,
                    "warmup_steps": 0,
                    "batch_size": 4,
                    "accumulate_steps": 1,
                    "learning_rate": 5e-6,
                    "minimum_learning_rate": 1e-5,
                }
            )
        }
    )
    return c


def test_same_state_loss_difference_and_next_update_restore(tmp_path):
    c = fixture(tmp_path)
    parent = make_arm(c, fresh_model(c))
    p = velocity_inputs(batch(2, 2, count=2)[0])
    update(parent, p)
    configs = sibling_configs(c)
    arms = {m: split_local(parent, config) for m, config in configs.items()}
    for arm in arms.values():
        ExperimentConfig.model_validate(arm["config"].model_dump())
        for step in (0, 20000, STEPS):
            np.testing.assert_allclose(arm["schedule"](step), 1e-5, rtol=1e-6)
        jax.tree.map(
            np.testing.assert_array_equal,
            lineart_state.payload({"x": parent}, 1),
            lineart_state.payload({"x": arm}, 1),
        )
    left = parent["gradient"](parent["model"], p)
    right = arms["control"]["gradient"](arms["control"]["model"], p)
    jax.tree.map(np.testing.assert_array_equal, left, right)
    results = {m: update(arm, p) for m, arm in arms.items()}
    assert results["edge"]["image_edge_energy"] > 0
    assert results["edge"]["gradient_norm"] > 0
    assert any(
        np.any(a != b)
        for a, b in zip(
            jax.tree.leaves(nnx.state(arms["control"]["model"], nnx.Param)),
            jax.tree.leaves(nnx.state(arms["edge"]["model"], nnx.Param)),
            strict=True,
        )
    )
    with checkpoints.manager(tmp_path / "checkpoints") as saved:
        lineart_state.save(saved, arms, 2)
        restored = {m: split_local(parent, config) for m, config in configs.items()}
        assert lineart_state.restore(saved, restored, 2) == 2
    for m in arms:
        assert update(arms[m], p) == update(restored[m], p)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 3),
        lineart_state.payload(restored, 3),
    )


def test_auxiliary_is_exactly_point_one_final_endpoint_energy(tmp_path):
    c = fixture(tmp_path)
    model = fresh_model(c)
    p = velocity_inputs(batch(3, 2, count=2)[0])
    configs = sibling_configs(c)
    values = {
        m: objective(model, p, config.loss, config.diffusion.min_noise, 0.1)[0]
        for m, config in configs.items()
    }
    u0 = tuple(jnp.ones_like(x) for x in p["images"])
    _, y0 = endpoints(model, p["images"], p["levels"], p["guides"], u0)
    middle = midpoint(p, y0, teacher=False)
    u1 = tuple(jnp.full_like(x, 0.5) for x in middle)
    raw1, _ = endpoints(model, middle, p["levels"], p["guides"], u1)
    expected = edge_energy_loss(raw1[0], p["target"], p["levels"][0] > c.diffusion.min_noise)
    np.testing.assert_allclose(values["edge"] - values["control"], 0.1 * expected, rtol=2e-5)


def test_both_arms_continue_exact_same_teacher_and_input_stream():
    c = load_config(
        "../jax-local-procedural-duration-20261012/runs/local-procedural-duration-20261012/configs/local.json"
    )
    source = load_config(
        "../jax-local-procedural-duration-20261012/runs/local-procedural-duration-20261012/configs/source.json"
    )
    preparation = prepare_batch(source)
    for index in (DATA_START, DATA_START + 1, DATA_START + 3999):
        expected, row = previous_batch(c, preparation, index, size=64)
        for config in sibling_configs(c).values():
            actual, actual_row = prepared_batch(config, preparation, index, size=64)
            jax.tree.map(np.testing.assert_array_equal, actual, expected)
            assert actual_row == row and row["data_step"] == index + 1
