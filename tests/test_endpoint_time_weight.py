import jax
import jax.numpy as jnp
import numpy as np
import pytest
from anchored_endpoint import noisy_gradient
from diagnose_learnability import update
from endpoint_time_weight import gradient, objective
from endpoint_time_weight_fit import run_sample, split_arms
from flax import nnx
from geometry_endpoint_fit import run_sample as original_sample
from test_progress_modulation import modulated

from lineartgen.training import checkpoints, lineart_state
from lineartgen.training.lineart_study import create_arm


class ConstantHead(nnx.Module):
    def __init__(self):
        self.bias = nnx.Param(jnp.array(0.0))

    def __call__(self, x, t, insert, s, u, return_network_output=False):
        assert return_network_output
        return jnp.full_like(x, self.bias[...])


@pytest.mark.parametrize("remaining", [1.0, 0.5, 0.1, 0.001])
def test_exact_time_weights_and_protected_exclusion(remaining):
    x = jnp.zeros((1, 16, 16, 1))
    t = jnp.ones_like(x).at[:, :, :8].set(0)
    y = jnp.full_like(x, 0.5).at[:, :, :8].set(-1)
    p = dict(
        images=(x,), levels=(t,), guides=(x,), progress=(jnp.full_like(x, remaining),), target=y
    )
    for floor in (1 / 32, 0.25, 1):
        (loss, metrics), g = nnx.value_and_grad(
            lambda m, f=floor: objective(m, p, f), has_aux=True
        )(ConstantHead())
        weight = 1 / max(remaining, floor) ** 2
        np.testing.assert_allclose(loss, 0.25 * weight, rtol=2e-6)
        np.testing.assert_allclose(g["bias"], -weight, rtol=2e-6)
        assert float(metrics["interval_endpoint_mse"]) == 0.25
        q = dict(p, levels=(jnp.zeros_like(x),))
        (loss, _), g = nnx.value_and_grad(
            lambda m, inputs=q, f=floor: objective(m, inputs, f), has_aux=True
        )(ConstantHead())
        assert float(loss) == 0 and float(g["bias"]) == 0


def test_original_floor_has_identical_actual_model_loss_and_gradient(tmp_path):
    _, model, _, p = modulated(tmp_path)
    actual = gradient(model, p, floor=1 / 32)
    old = noisy_gradient(model, p)
    jax.tree.map(np.testing.assert_array_equal, actual, old)


def test_independent_full_state_restore_and_next_update(tmp_path):
    _, model, c, p = modulated(tmp_path)
    original = create_arm(c, model)
    original["gradient"] = noisy_gradient
    update(original, p)
    arms = split_arms(original, {m: c for m in ("cap", "endpoint")})
    before = jax.tree.map(lambda x: np.asarray(x).copy(), lineart_state.payload({"x": original}, 1))
    for arm in arms.values():
        jax.tree.map(np.testing.assert_array_equal, before, lineart_state.payload({"x": arm}, 1))
    for method in ("cap", "endpoint"):
        stats = update(arms[method], p)
        assert stats["gradient_norm"] > 0 and np.isfinite(stats["loss"])
        jax.tree.map(
            np.testing.assert_array_equal, before, lineart_state.payload({"x": original}, 1)
        )
        if method == "cap":
            jax.tree.map(
                np.testing.assert_array_equal,
                before,
                lineart_state.payload({"x": arms["endpoint"]}, 1),
            )
    with checkpoints.manager(tmp_path / "state") as saved:
        lineart_state.save(saved, arms, 2)
        restored = split_arms(original, {m: c for m in arms})
        assert lineart_state.restore(saved, restored, 2) == 2
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 2),
        lineart_state.payload(restored, 2),
    )
    for method in arms:
        assert update(arms[method], p) == update(restored[method], p)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 3),
        lineart_state.payload(restored, 3),
    )


@pytest.mark.parametrize("method", ["control", "cap", "endpoint"])
def test_sampler_is_unchanged_and_protected_exact(tmp_path, method):
    _, model, _, p = modulated(tmp_path)
    s, t = p["guides"][0], p["levels"][0]
    key = jax.random.PRNGKey(18)
    output = run_sample(model, s, t, key, 2, method)
    np.testing.assert_array_equal(output, original_sample(model, s, t, key, 2, "modulated"))
    mask = np.asarray(t) == 0
    np.testing.assert_array_equal(output[mask], np.asarray(s)[mask])
