import jax
import jax.numpy as jnp
import numpy as np
import pytest
from diagnose_learnability import update
from flax import nnx
from interval_velocity_fit import interval_error, interval_objective, split_arms
from test_velocity_bridge_toy import fixture
from velocity_bridge_toy import make_arms as original_arms

from lineartgen.training import checkpoints, lineart_state


def test_interval_error_correct_saturation_wrong_saturation_and_antialias_gradients():
    clean = jnp.array([1.0, 1.0, -1.0, -1.0, 0.3, 0.3])
    prediction = jnp.array([0.2, -0.2, -0.2, 0.2, 0.2, -0.2])
    expected = jnp.array([0.0, -0.2, 0.0, 0.2, 0.2, -0.2])
    np.testing.assert_array_equal(
        interval_error(prediction, jnp.zeros_like(prediction), clean), expected
    )
    grad = jax.grad(lambda v: jnp.sum(interval_error(v, jnp.zeros_like(v), clean) ** 2))(prediction)
    np.testing.assert_allclose(grad, [0.0, -0.4, 0.0, 0.4, 0.4, -0.4], atol=1e-7)


@pytest.mark.parametrize("remaining", [1.0, 0.5, 0.001])
def test_velocity_interval_is_time_normalized_endpoint_interval(remaining):
    clean = np.array([1.0, -1.0, 0.25, 1.0, -1.0, -0.25])
    z = np.array([0.4, -0.3, 0.5, 0.9, -0.7, 0.1])
    v = np.array([0.9, -0.9, 0.2, -0.2, 0.2, -0.6])
    target = clean - z
    residual = np.asarray(interval_error(jnp.asarray(v), jnp.asarray(target), jnp.asarray(clean)))
    x = remaining * z + (1 - remaining) * clean
    endpoint = x + remaining * v
    error = endpoint - clean
    physical = np.where(
        clean == 1, np.minimum(error, 0), np.where(clean == -1, np.maximum(error, 0), error)
    )
    np.testing.assert_allclose(residual * remaining, physical, atol=1e-7, rtol=1e-6)


class ConstantModel(nnx.Module):
    def __init__(self):
        self.value = nnx.Param(jnp.array(0.2))

    def __call__(self, x, t, insert, s, u, return_preclip=False):
        raw = x + self.value[...]
        return jnp.clip(raw, -1, 1), raw


def test_fully_protected_interval_loss_and_gradients_are_zero():
    x = jnp.ones((2, 16, 16, 1))
    p = dict(
        images=(x,),
        levels=(jnp.zeros_like(x),),
        guides=(x,),
        progress=(x,),
        target=x,
        velocity_target=-x,
    )
    model = ConstantModel()
    assert float(interval_objective(model, p)[0]) == 0
    grads = nnx.grad(lambda m: interval_objective(m, p)[0])(model)
    assert float(grads["value"]) == 0


def test_split_states_preserve_mse_update_and_resume_each_independently(tmp_path):
    _, p, _ = fixture(tmp_path)
    reference = original_arms(tmp_path)["velocity"]
    for _ in range(2):
        update(reference, p)
    config = reference["config"]
    c = config.model_copy(
        update={
            "train": config.train.model_copy(
                update={"steps": 36000, "learning_rate_decay_steps": config.train.steps}
            )
        }
    )
    configs = {"mse": c, "interval": c}
    arms = split_arms(reference, configs)
    initial = jax.tree.map(
        lambda v: np.asarray(v).copy(), lineart_state.payload({"interval": reference}, 2)
    )
    for name in arms:
        jax.tree.map(
            np.testing.assert_array_equal,
            lineart_state.payload({name: arms[name]}, 2),
            lineart_state.payload({name: reference}, 2),
        )
        for n in (0, 5999, 6000, 30000, 36000):
            assert float(arms[name]["schedule"](n)) == float(reference["schedule"](n))
    mse = update(arms["mse"], p)
    jax.tree.map(
        np.testing.assert_array_equal,
        initial,
        lineart_state.payload({"interval": arms["interval"]}, 2),
    )
    jax.tree.map(
        np.testing.assert_array_equal, initial, lineart_state.payload({"interval": reference}, 2)
    )
    assert mse == update(reference, p)
    interval = update(arms["interval"], p)
    assert interval["gradient_norm"] > 0 and interval["loss"] <= mse["loss"]
    with checkpoints.manager(tmp_path / "interval-checkpoints") as saved:
        lineart_state.save(saved, arms, 3)
        restored = split_arms(reference, configs)
        assert lineart_state.restore(saved, restored) == 3
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 3),
        lineart_state.payload(restored, 3),
    )
    for name in arms:
        assert update(arms[name], p) == update(restored[name], p)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 4),
        lineart_state.payload(restored, 4),
    )
