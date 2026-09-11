import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from scipy.ndimage import gaussian_filter

from lineartgen.config import ExperimentConfig, LossConfig, SketchAugmentationConfig
from lineartgen.diffusion import noise_map, pyramid, training_inputs
from lineartgen.models import create_model
from lineartgen.sketch import augment_sketch, gaussian_blur, restoration_target
from lineartgen.training.comparison import comparison_configs
from lineartgen.training.losses import image_loss
from lineartgen.training.steps import make_steps


def settings():
    return ExperimentConfig.model_validate(
        {
            "model": {
                "architecture": "nested",
                "channels": [2, 3],
                "nested_channels": [[3, 4], [4, 5]],
                "convolutions_per_block": 1,
                "output_mode": "residual",
            },
            "matching": {"method": "independent", "pool_size": 4},
            "diffusion": {"pattern_weights": [0, 0, 1, 0], "identity_probability": 0},
            "sketch": {"probability": 1},
            "loss": {"noise_weight_floor": 0.05, "contrast_active_only": True},
        }
    )


@pytest.mark.parametrize("mode", ["none", "tone", "blur", "strokes"])
def test_sketch_preserves_white_and_is_deterministic_and_bounded(mode):
    config = SketchAugmentationConfig(mode=mode, probability=1)
    apply = jax.jit(lambda image, key: augment_sketch(image, key, config))
    key = jax.random.PRNGKey(8)
    white = jnp.ones((2, 32, 32, 1))
    np.testing.assert_array_equal(apply(white, key), white)
    clean = white.at[:, 10:20, 15:17].set(-1)
    first = np.asarray(apply(clean, key))
    np.testing.assert_array_equal(first, apply(clean, key))
    assert np.isfinite(first).all() and first.min() >= -1 and first.max() <= 1
    if mode == "none":
        np.testing.assert_array_equal(first, clean)
    else:
        assert not np.array_equal(first, clean)
        assert not np.array_equal(first, apply(clean, jax.random.PRNGKey(9)))


def test_blur_matches_independent_reference_and_zero_sigma_is_identity():
    ink = jnp.zeros((2, 32, 32, 1)).at[:, 16, 16].set(1)
    actual = gaussian_blur(ink, jnp.array([0.0, 1.25]), 2)
    np.testing.assert_array_equal(actual[0], ink[0])
    expected = gaussian_filter(np.asarray(ink[1, ..., 0]), sigma=1.25, radius=6, mode="constant")
    np.testing.assert_allclose(actual[1, ..., 0], expected, atol=2e-8)
    np.testing.assert_allclose(actual[1].sum(), 1, atol=1e-6)


def test_disabled_sketch_keeps_existing_training_inputs_exact():
    config = settings()
    clean = jax.random.uniform(jax.random.PRNGKey(0), (4, 32, 32, 1))
    key = jax.random.PRNGKey(1)
    old = training_inputs(clean, key, config.diffusion, config.matching, levels=3)
    for sketch in (
        SketchAugmentationConfig(),
        SketchAugmentationConfig(mode="strokes", probability=0),
    ):
        actual = training_inputs(
            clean, key, config.diffusion, config.matching, levels=3, sketch=sketch
        )
        jax.tree.map(np.testing.assert_array_equal, old, actual)


def test_rough_input_is_downsampled_before_common_noise_and_protected_teacher_is_rough():
    variants = comparison_configs(settings(), study="sketch")
    clean = jnp.ones((4, 32, 32, 1)).at[:, :, 14:18].set(-1)
    key = jax.random.PRNGKey(7)
    map_key, noise_key = jax.random.split(key)
    expected_mask = noise_map(clean, map_key, settings().diffusion)
    for config in variants.values():
        source = augment_sketch(clean, jax.random.fold_in(key, 0x534B4554), config.sketch)
        expected_target = restoration_target(clean, source, expected_mask)
        actual = jax.jit(
            lambda x, config=config: training_inputs(
                x, key, config.diffusion, config.matching, levels=3, sketch=config.sketch
            )
        )(clean)
        jax.tree.map(np.testing.assert_array_equal, actual.noise_levels, pyramid(expected_mask, 3))
        jax.tree.map(
            lambda a, b: np.testing.assert_allclose(a, b, atol=3e-7),
            actual.targets,
            pyramid(expected_target, 3),
        )
        for src, strength, rng, noisy in zip(
            pyramid(source, 3),
            actual.noise_levels,
            jax.random.split(noise_key, 3),
            actual.noisy,
            strict=True,
        ):
            expected = src * jnp.sqrt(1 - strength) + jax.random.normal(rng, src.shape) * jnp.sqrt(
                strength
            )
            np.testing.assert_allclose(noisy, expected, atol=5e-7)
        mask = np.asarray(expected_mask) == 0
        np.testing.assert_allclose(
            np.asarray(actual.noisy[0])[mask], np.asarray(source)[mask], atol=3e-7
        )
        np.testing.assert_allclose(
            np.asarray(actual.targets[0])[mask], np.asarray(source)[mask], atol=3e-7
        )


def test_low_noise_loss_floor_and_gray_protected_contrast():
    config = LossConfig(
        balance_tones=False,
        edge_weight=0,
        contrast_weight=0,
        noise_weight_floor=0.05,
        contrast_active_only=True,
    )
    predicted = jnp.ones((1, 4, 4, 1))
    clean = jnp.zeros_like(predicted)
    _, metrics = image_loss(predicted, clean, jnp.full_like(clean, 0.002), config, 0.001)
    np.testing.assert_allclose(metrics["denoise"], 20)
    _, metrics = image_loss(clean, clean, jnp.zeros_like(clean), config, 0.001)
    assert float(metrics["contrast"]) == float(metrics["denoise"]) == 0
    assert all(np.isfinite(v) for v in metrics.values())


def test_augmented_training_has_finite_nonzero_gradients():
    config = comparison_configs(settings(), study="sketch")["strokes"]
    model = create_model(config.model, rngs=nnx.Rngs(42))
    clean = jnp.ones((2, 16, 16, 1)).at[:, :, 7:9].set(-1)
    inputs = training_inputs(
        clean,
        jax.random.PRNGKey(3),
        config.diffusion,
        config.matching,
        levels=2,
        sketch=config.sketch,
    )
    metrics, gradients = make_steps(config)[0](model, inputs)
    assert np.isfinite(float(metrics["loss"]))
    assert all(np.isfinite(v).all() for v in jax.tree.leaves(gradients))
    assert any(np.any(v) for v in jax.tree.leaves(gradients))


def test_invalid_sketch_settings_and_paired_masks_are_rejected():
    with pytest.raises(ValueError, match="independent"):
        ExperimentConfig.model_validate({"sketch": {"mode": "blur"}})
    with pytest.raises(ValueError):
        SketchAugmentationConfig(opacity_min=0.9, opacity_max=0.3)
    config = settings()
    with pytest.raises(ValueError, match="image-independent"):
        comparison_configs(
            config.model_copy(
                update={
                    "diffusion": config.diffusion.model_copy(
                        update={"pattern_weights": (1, 1, 1, 1)}
                    )
                }
            ),
            study="sketch",
        )
