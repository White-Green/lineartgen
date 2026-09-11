import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from lineartgen.config import ExperimentConfig
from lineartgen.diffusion import pyramid, recursive_forward_all, sample, training_inputs
from lineartgen.export import export_model
from lineartgen.models import create_model
from lineartgen.models.serialization import load_inference, save_inference
from lineartgen.noise_matching import assignment_indices, pairwise_noise_costs
from lineartgen.training.comparison import comparison_configs
from lineartgen.training.steps import make_steps


def settings():
    return ExperimentConfig.model_validate(
        {
            "model": {
                "architecture": "nested",
                "channels": [2, 3],
                "nested_channels": [[3, 4], [4, 5]],
                "convolutions_per_block": 1,
            },
            "train": {"batch_size": 2, "accumulate_steps": 2},
            "matching": {"pool_size": 8},
            "diffusion": {"identity_probability": 0, "pattern_weights": [1, 0, 0, 0]},
        }
    )


def test_methods_preserve_initial_parameters_and_optimizer_conditions():
    variants = comparison_configs(settings(), study="methods")
    states = []
    for config in variants.values():
        states.append(nnx.state(create_model(config.model, rngs=nnx.Rngs(42)), nnx.Param))
        assert config.train == settings().train
        assert config.diffusion == settings().diffusion
        assert config.loss == settings().loss
    for state in states[1:]:
        jax.tree.map(np.testing.assert_array_equal, states[0], state)


def test_assignment_variants_use_the_same_raw_noise_candidates():
    configs = comparison_configs(settings(), study="methods")
    clean = jax.random.uniform(jax.random.PRNGKey(3), (4, 16, 16, 1), minval=-1, maxval=1)
    key = jax.random.PRNGKey(4)
    prepared = {
        name: training_inputs(clean, key, c.diffusion, c.matching, levels=2)
        for name, c in configs.items()
    }
    jax.tree.map(np.testing.assert_array_equal, prepared["current"], prepared["residual"])
    noise_key = jax.random.split(key)[1]
    for level, rng in enumerate(jax.random.split(noise_key, 2)):
        target = prepared["current"].targets[level]
        strength = prepared["current"].noise_levels[level]
        raw = jax.random.normal(rng, (8, *target.shape[1:]))
        for name, count in [("current", 8), ("ot_full", 4), ("independent_noise", 4)]:
            candidates = raw[:count]
            if name != "independent_noise":
                indices = assignment_indices(pairwise_noise_costs(target, strength, candidates))
                if name == "ot_full":
                    np.testing.assert_array_equal(np.sort(indices), np.arange(4))
                candidates = candidates[indices]
            expected = target * jnp.sqrt(1 - strength) + candidates * jnp.sqrt(strength)
            np.testing.assert_allclose(prepared[name].noisy[level], expected, atol=1e-6)


@pytest.mark.parametrize("levels", [1, 3])
def test_residual_preserves_protected_pixels_and_autonomous_prefixes(levels):
    config = comparison_configs(settings(), study="methods")["residual"]
    model = create_model(config.model, rngs=nnx.Rngs(42))
    image = jax.random.uniform(jax.random.PRNGKey(1), (1, 16, 16, 1), minval=-1, maxval=1)
    strength = jnp.full_like(image, 0.5).at[:, :, :8].set(0)
    images, noises = pyramid(image, levels), pyramid(strength, levels)
    before = nnx.jit(recursive_forward_all)(model, images, noises)
    for index in range(3):
        subnet = model.extract(index)
        actual = nnx.jit(recursive_forward_all)(subnet, images, noises)[index]
        jax.tree.map(
            lambda a, b: np.testing.assert_allclose(a, b, atol=2e-6), before[index], actual
        )
        for prediction, target, noise in zip(actual, images, noises, strict=True):
            protected = np.asarray(noise) == 0
            np.testing.assert_array_equal(
                np.asarray(prediction)[protected], np.asarray(target)[protected]
            )
        for noise in (strength, jnp.zeros_like(strength)):
            output = nnx.jit(
                lambda m, n: sample(
                    m, image, jax.random.PRNGKey(5), strength=n, steps=5, levels=levels
                )
            )(subnet, noise)
            protected = np.asarray(noise) == 0
            np.testing.assert_array_equal(
                np.asarray(output)[protected], np.asarray(image)[protected]
            )
    altered = nnx.clone(model)
    altered.heads[2].layers[-1].bias[...] += 0.2
    after = nnx.jit(recursive_forward_all)(altered, images, noises)
    jax.tree.map(np.testing.assert_array_equal, before[:2], after[:2])


def test_residual_has_finite_nonzero_training_gradients_and_roundtrips(tmp_path):
    config = comparison_configs(settings(), study="methods")["residual"]
    model = create_model(config.model, rngs=nnx.Rngs(42))
    clean = jnp.full((4, 16, 16, 1), 0.5)
    prepared = training_inputs(
        clean, jax.random.PRNGKey(3), config.diffusion, config.matching, levels=2
    )
    metrics, gradients = make_steps(config)[0](model, prepared)
    assert np.isfinite(float(metrics["loss"]))
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(gradients))
    assert any(np.any(x) for x in jax.tree.leaves(gradients))
    save_inference(model, config, tmp_path / "weights", provenance="test")
    restored, saved = load_inference(tmp_path / "weights")
    assert saved.model.output_mode == "residual"
    jax.tree.map(
        np.testing.assert_array_equal,
        recursive_forward_all(model, prepared.noisy, prepared.noise_levels),
        recursive_forward_all(restored, prepared.noisy, prepared.noise_levels),
    )
    ordinary, ordinary_config = model.extract(0).to_unet()
    assert ordinary_config.output_mode == "residual"
    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(a, b, atol=2e-6),
        recursive_forward_all(model, prepared.noisy, prepared.noise_levels)[0],
        recursive_forward_all(ordinary, prepared.noisy, prepared.noise_levels)[0],
    )


def test_residual_dynamic_onnx_export(tmp_path):
    pytest.importorskip("jax2onnx")
    config = comparison_configs(settings(), study="methods")["residual"]
    model = create_model(config.model, rngs=nnx.Rngs(42)).extract(0)
    assert export_model(model, config, tmp_path / "residual.onnx").is_file()
