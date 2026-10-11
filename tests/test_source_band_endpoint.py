import jax
import jax.numpy as jnp
import numpy as np
import pytest
from anchored_endpoint import noisy_gradient
from diagnose_learnability import update
from geometry_endpoint_fit import run_sample as parent_sample
from scipy.ndimage import maximum_filter
from source_band_endpoint import observed_band, risk
from source_band_fit import run_sample, split_arms
from test_progress_modulation import modulated

from lineartgen.training import checkpoints, lineart_state
from lineartgen.training.lineart_study import create_arm


def test_band_matches_spatial_dilation_and_excludes_protected_pixels():
    alpha = np.random.default_rng(28).uniform(0, 0.3, (2, 19, 23, 1)).astype(np.float32)
    source = 1 - 2 * alpha
    levels = np.full_like(source, 0.5)
    levels[:, :, :9] = 0
    expected = maximum_filter(
        np.clip((1 - source) / 2, 0, 1), size=(1, 9, 9, 1), mode="constant", cval=0
    )
    expected[levels == 0] = 0
    np.testing.assert_array_equal(observed_band(jnp.asarray(source), jnp.asarray(levels)), expected)


@pytest.mark.parametrize("all_protected", [False, True])
def test_blank_falls_back_to_active_mean_and_protected_errors_have_zero_gradient(all_protected):
    source = jnp.ones((2, 16, 16, 1))
    levels = (
        jnp.zeros_like(source) if all_protected else jnp.full_like(source, 0.5).at[:, :, :8].set(0)
    )
    error = jnp.arange(source.size, dtype=jnp.float32).reshape(source.shape) / source.size
    actual, base, local = risk(error**2, source, levels)
    np.testing.assert_array_equal(actual, base)
    np.testing.assert_array_equal(local, base)
    gradient = jax.grad(lambda e: risk(e**2, source, levels)[0])(error)
    assert np.isfinite(gradient).all()
    assert np.all(np.asarray(gradient)[np.asarray(levels) == 0] == 0)
    if all_protected:
        assert float(actual) == 0


def test_sparse_observed_ink_gets_more_weight_and_uniform_opacity_scaling_cancels():
    alpha = jnp.zeros((1, 32, 32, 1)).at[:, 16, 16].set(0.25)
    source = 1 - 2 * alpha
    levels = jnp.full_like(source, 0.5)
    error = jnp.ones_like(source)
    grad = jax.grad(lambda e: risk(e**2, source, levels)[0])(error)
    assert float(grad[0, 16, 16, 0]) > 5 * float(grad[0, 0, 0, 0])
    gradient_scaled = jax.grad(lambda e: risk(e**2, 1 - alpha, levels)[0])(error)
    np.testing.assert_array_equal(grad, gradient_scaled)


def test_source_only_weights_leave_conditional_pointwise_minimum_unchanged():
    source = jnp.ones((1, 16, 16, 1)).at[:, 7:9, 7:9].set(0.5)
    levels = jnp.full_like(source, 0.5)
    a = jnp.full_like(source, -0.6)
    b = jnp.full_like(source, 0.8)
    prob = 0.3
    mean = prob * a + (1 - prob) * b

    def expected(pred):
        return (
            prob * risk((pred - a) ** 2, source, levels)[0]
            + (1 - prob) * risk((pred - b) ** 2, source, levels)[0]
        )

    np.testing.assert_allclose(jax.grad(expected)(mean), 0, atol=2e-9, rtol=0)
    assert float(expected(mean + 0.05)) > float(expected(mean))
    assert float(expected(mean - 0.05)) > float(expected(mean))


def test_independent_optimizer_copies_and_exact_next_update_restore(tmp_path):
    _, model, c, p = modulated(tmp_path)
    original = create_arm(c, model)
    original["gradient"] = noisy_gradient
    update(original, p)
    configs = {m: c for m in ("control", "band")}
    arms = split_arms(original, configs)
    before = jax.tree.map(lambda x: np.asarray(x).copy(), lineart_state.payload({"x": original}, 1))
    for arm in arms.values():
        jax.tree.map(np.testing.assert_array_equal, before, lineart_state.payload({"x": arm}, 1))
    control_metrics = update(arms["control"], p)
    assert control_metrics == update(original, p)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload({"x": original}, 2),
        lineart_state.payload({"x": arms["control"]}, 2),
    )
    jax.tree.map(
        np.testing.assert_array_equal, before, lineart_state.payload({"x": arms["band"]}, 1)
    )
    metrics = update(arms["band"], p)
    assert metrics["gradient_norm"] > 0 and np.isfinite(metrics["loss"])
    with checkpoints.manager(tmp_path / "state") as saved:
        lineart_state.save(saved, arms, 2)
        restored = split_arms(original, configs)
        assert lineart_state.restore(saved, restored, 2) == 2
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 2),
        lineart_state.payload(restored, 2),
    )
    for m in arms:
        assert update(arms[m], p) == update(restored[m], p)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 3),
        lineart_state.payload(restored, 3),
    )


@pytest.mark.parametrize("method", ["initial", "control", "band"])
def test_all_methods_keep_the_original_endpoint_sampler_and_protection(tmp_path, method):
    _, model, _, p = modulated(tmp_path)
    source, levels = p["guides"][0], p["levels"][0]
    key = jax.random.PRNGKey(33)
    value = run_sample(model, source, levels, key, 2, method)
    np.testing.assert_array_equal(value, parent_sample(model, source, levels, key, 2, "modulated"))
    mask = np.asarray(levels) == 0
    np.testing.assert_array_equal(value[mask], np.asarray(source)[mask])
