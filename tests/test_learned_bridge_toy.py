import jax
import jax.numpy as jnp
import numpy as np
import pytest
from diagnose_learnability import update
from flax import nnx
from learned_bridge_toy import (
    batch,
    config,
    initialize,
    make_arms,
    measure,
    path_inputs,
    sample,
    targets,
)

from lineartgen.config import ModelConfig, save_config
from lineartgen.diffusion import downsample, pyramid
from lineartgen.models import create_model
from lineartgen.models.serialization import save_inference
from lineartgen.training import checkpoints, lineart_state


def test_same_source_and_balanced_labels_with_identical_starting_states():
    prepared, labels, tau = batch(8, 3)
    assert np.count_nonzero(np.asarray(labels) == 0) == 8
    zero_time = np.asarray(tau[:, 0, 0, 0] == 0)
    for a, b in zip(prepared["direct"]["images"], prepared["bridge"]["images"], strict=True):
        np.testing.assert_array_equal(np.asarray(a)[zero_time], np.asarray(b)[zero_time])
    for key in ("levels", "guides", "target"):
        jax.tree.map(
            np.testing.assert_array_equal, prepared["direct"][key], prepared["bridge"][key]
        )
    repeated, repeated_labels, repeated_tau = batch(8, 3)
    jax.tree.map(np.testing.assert_array_equal, prepared, repeated)
    np.testing.assert_array_equal(labels, repeated_labels)
    np.testing.assert_array_equal(tau, repeated_tau)


def test_noise_is_drawn_after_reduction_and_independent_between_scales():
    source = jnp.ones((16, 64, 64, 1))
    strength = jnp.full_like(source, 0.5)
    key = jax.random.PRNGKey(4)
    images, levels, guides = initialize(source, strength, key, 3)
    keys = jax.random.split(key, 3)
    for x, t, s, k in zip(images, levels, guides, keys, strict=True):
        np.testing.assert_array_equal(x, s + 0.25 * t * jax.random.normal(k, s.shape))
        np.testing.assert_allclose(np.asarray(x - s).std(), 0.125, rtol=0.04)
    assert not np.allclose(np.asarray(images[1]), np.asarray(downsample(images[0])))


def test_time_one_and_protected_path_values_exact():
    a, b, s = map(lambda x: jnp.asarray(x[None]), targets())
    t = jnp.full_like(s, 0.5).at[:, :, :32].set(0)
    value = path_inputs(s, a, t, jax.random.PRNGKey(2), jnp.ones((1, 1, 1, 1)), 3)
    expected = pyramid(jnp.where(t > 0, a, s), 3)
    jax.tree.map(np.testing.assert_array_equal, value["images"], expected)
    for x, mask, guide in zip(value["images"], value["levels"], value["guides"], strict=True):
        np.testing.assert_array_equal(
            np.asarray(x)[np.asarray(mask) == 0], np.asarray(guide)[np.asarray(mask) == 0]
        )


@pytest.mark.parametrize("steps", [1, 8, 32])
def test_sampler_fixed_endpoint_and_protection(steps):
    a, b, s = map(lambda x: jnp.asarray(x[None]), targets())
    t = jnp.full_like(s, 0.5).at[:, :, :32].set(0)
    teacher = jnp.where(t > 0, a, s)
    result = sample(
        lambda x, t, s, u: pyramid(teacher, 3), s, t, jax.random.PRNGKey(2), depth=3, steps=steps
    )
    np.testing.assert_allclose(result, teacher, atol=1e-7, rtol=0)
    np.testing.assert_array_equal(np.asarray(result)[:, :, :32], np.asarray(s)[:, :, :32])
    with pytest.raises(ValueError):
        sample(None, s, t, jax.random.PRNGKey(2), depth=3, steps=0)


def test_two_endpoint_oracle_resolves_modes_and_mean_has_lower_paired_mse():
    a, b, s = targets()
    source = jnp.broadcast_to(s, (64, *s.shape))
    t = jnp.full_like(source, 0.5)
    mean, half = map(jnp.asarray, ((a + b) / 2, (a - b) / 2))

    def oracle(x, levels, guides, progress):
        tau = 1 - progress[0]
        center = (1 - tau) * guides[0] + tau * mean
        variance = (0.25 * levels[0] * (1 - tau)) ** 2
        logit = jnp.sum((x[0] - center) * half / variance, axis=(1, 2, 3), keepdims=True) * tau
        return (mean + half * jnp.tanh(logit),)

    z = sample(oracle, source, t, jax.random.PRNGKey(7), depth=1, steps=32)
    direct = sample(oracle, source, t, jax.random.PRNGKey(7), depth=1, steps=1)
    metrics = measure(np.asarray(z), a, b, np.asarray(source), np.asarray(t))
    assert metrics["mean_relative_endpoint_rms"] < 0.01
    assert 6 < metrics["nearest_A"] < 58
    m = measure(np.asarray(direct), a, b, np.asarray(source), np.asarray(t))
    np.testing.assert_allclose(m["mean_relative_endpoint_rms"], 1, atol=1e-14, rtol=0)
    assert m["teacher_averaged_mse"] < metrics["teacher_averaged_mse"]


def test_real_update_separate_arms_full_state_and_next_update_restore(tmp_path):
    c = config()
    c = c.model_copy(
        update={
            "model": ModelConfig(
                channels=(2, 4),
                convolutions_per_block=2,
                detail_channels=4,
                output_mode="residual",
                source_conditioning=True,
                progress_conditioning=True,
            ),
            "train": c.train.model_copy(update={"batch_size": 2, "warmup_steps": 0}),
        }
    )
    model = create_model(c.model, rngs=nnx.Rngs(42))
    save_inference(model, c, tmp_path / "initial", provenance="test")
    (tmp_path / "configs").mkdir()
    for name in ("direct", "bridge"):
        save_config(c, tmp_path / f"configs/{name}.json")
    arms = make_arms(tmp_path)
    prepared, _, _ = batch(1, 2, count=2)
    # This random fixture happens to draw time zero twice. Explicitly exercise
    # nonzero time when checking that the progress projections can learn.
    direct = prepared["direct"]
    prepared["bridge"] = path_inputs(
        direct["guides"][0],
        direct["target"],
        direct["levels"][0],
        jax.random.PRNGKey(19),
        jnp.full((2, 1, 1, 1), 0.5),
        2,
    )
    before = jax.tree.map(
        lambda x: np.asarray(x).copy(), nnx.state(arms["direct"]["model"], nnx.Param)
    )
    result = update(arms["bridge"], prepared["bridge"])
    assert result["gradient_norm"] > 0 and np.isfinite(result["loss"])
    jax.tree.map(
        np.testing.assert_array_equal, before, nnx.state(arms["direct"]["model"], nnx.Param)
    )
    progress = nnx.state(arms["bridge"]["model"], nnx.Param)["progress_head"]
    assert np.any(np.asarray(progress["kernel"]) != 0)
    update(arms["direct"], prepared["direct"])
    with checkpoints.manager(tmp_path / "checkpoints") as saved:
        lineart_state.save(saved, arms, 1)
        restored = make_arms(tmp_path)
        assert lineart_state.restore(saved, restored) == 1
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 1),
        lineart_state.payload(restored, 1),
    )
    for name in arms:
        assert update(arms[name], prepared[name]) == update(restored[name], prepared[name])
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 2),
        lineart_state.payload(restored, 2),
    )
