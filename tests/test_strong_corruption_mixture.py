import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lineartgen.config import SketchAugmentationConfig
from lineartgen.sketch import augment_sketch


def fixture():
    image = jnp.ones((4, 64, 64, 1))
    image = image.at[:, 12:52, 30:33].set(-1)
    return image.at[1, 30:33, 12:52].set(-1).at[2].set(1).at[3].set(-1)


@pytest.mark.parametrize("mode", ["tone", "blur", "strokes", "mixed"])
def test_zero_strong_probability_matches_frozen_parent_bitwise(mode):
    path = (
        Path(__file__).resolve().parents[2]
        / "jax-source-interval-l1-20261010/src/lineartgen/sketch.py"
    )
    spec = importlib.util.spec_from_file_location("frozen_sketch", path)
    parent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parent)
    config = SketchAugmentationConfig(mode=mode, probability=0.8)
    key = jax.random.PRNGKey(91)
    current = jax.jit(lambda x, k: augment_sketch(x, k, config))
    legacy = jax.jit(lambda x, k: parent.augment_sketch(x, k, config))
    np.testing.assert_array_equal(current(fixture(), key), legacy(fixture(), key))


@pytest.mark.parametrize("mode", ["tone", "blur", "strokes", "mixed"])
def test_per_image_branch_selection_preserves_original_draws(mode):
    config = SketchAugmentationConfig(mode=mode, probability=0.8, strong_probability=0.5)
    ordinary = config.model_copy(update={"strong_probability": 0})
    strong = ordinary.model_copy(
        update={
            "opacity_min": 0.2,
            "opacity_max": 0.45,
            "blur_sigma_min": 1.5,
            "blur_sigma_max": 2.5,
            "displacement": 3,
        }
    )
    image = fixture()
    actual = jax.jit(lambda x, k: augment_sketch(x, k, config))
    old = jax.jit(lambda x, k: augment_sketch(x, k, ordinary))
    hard = jax.jit(lambda x, k: augment_sketch(x, k, strong))
    choices = []
    enabled_draws = []
    for seed in range(4):
        key = jax.random.PRNGKey(seed)
        choose = jax.random.bernoulli(jax.random.fold_in(key, 5), 0.5, (4, 1, 1, 1))
        enabled = jax.random.bernoulli(jax.random.fold_in(key, 0), 0.8, (4, 1, 1, 1))
        choices.extend(np.asarray(choose).ravel())
        enabled_draws.extend(np.asarray(enabled).ravel())
        expected = jnp.where(choose, hard(image, key), old(image, key))
        np.testing.assert_allclose(actual(image, key), expected, atol=1e-6, rtol=0)
        inactive = np.broadcast_to(~np.asarray(enabled), image.shape)
        np.testing.assert_array_equal(
            np.asarray(actual(image, key))[inactive], np.asarray(image)[inactive]
        )
    assert any(choices) and not all(choices) and not all(enabled_draws)


def test_probability_endpoints_and_disabled_augmentation():
    config = SketchAugmentationConfig(mode="mixed", probability=0.8, strong_probability=1)
    strong = config.model_copy(
        update={
            "strong_probability": 0,
            "opacity_min": 0.2,
            "opacity_max": 0.45,
            "blur_sigma_min": 1.5,
            "blur_sigma_max": 2.5,
            "displacement": 3,
        }
    )
    image, key = fixture(), jax.random.PRNGKey(17)
    np.testing.assert_allclose(
        augment_sketch(image, key, config), augment_sketch(image, key, strong), atol=1e-6, rtol=0
    )
    for changes in ({"mode": "none"}, {"probability": 0}):
        np.testing.assert_array_equal(
            augment_sketch(image, key, config.model_copy(update=changes)), image
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"strong_probability": -1},
        {"strong_probability": 1.1},
        {"strong_opacity_min": 0.5, "strong_opacity_max": 0.2},
        {"strong_blur_sigma_min": 3, "strong_blur_sigma_max": 2},
        {"strong_displacement": 9},
    ],
)
def test_invalid_strong_ranges_rejected(changes):
    with pytest.raises(ValueError):
        SketchAugmentationConfig(**changes)
