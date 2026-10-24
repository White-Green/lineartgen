import jax.numpy as jnp
import numpy as np
import pytest
from lineartgen.models.masked_normalization import normalized_masked_guide as training
from lineartgen_runtime.graph import normalized_guide
from lineartgen_runtime.masked_normalization import normalized_masked_guide as runtime


@pytest.mark.parametrize("noise", [0.0, 0.1, 0.5, 1.0])
def test_uniform_protection_matches_existing_preprocessor(noise):
    rng = np.random.default_rng(20261021)
    source = rng.uniform(-1, 1, (2, 35, 49, 1)).astype(np.float32)
    levels = np.full_like(source, noise)
    np.testing.assert_array_equal(runtime(source, levels), normalized_guide(source))
    np.testing.assert_allclose(
        training(jnp.asarray(source), jnp.asarray(levels)),
        runtime(source, levels),
        atol=1e-7,
        rtol=1e-6,
    )


def test_protected_dark_ink_does_not_attenuate_continuing_faint_stroke():
    ink = np.zeros((1, 64, 96, 1), np.float32)
    ink[:, 28:36, :48] = 1
    ink[:, 28:36, 48:] = 0.4
    source = 1 - 2 * ink
    noise = np.zeros_like(source)
    noise[:, :, 48:] = 0.5
    original = normalized_guide(source)
    actual = runtime(source, noise)
    assert original[0, 32, 48, 0] > 0.6
    np.testing.assert_allclose(actual[0, 32, :, 0], 0.2, atol=1e-7)
    np.testing.assert_allclose(
        training(jnp.asarray(source), jnp.asarray(noise)), actual, atol=1e-7, rtol=1e-6
    )


def test_no_ink_is_added_across_a_protection_boundary():
    source = np.ones((1, 64, 96, 1), np.float32)
    source[:, 28:36, :48] = -1
    noise = np.zeros_like(source)
    noise[:, :, 48:] = 0.5
    actual = runtime(source, noise)
    np.testing.assert_array_equal(actual[:, :, 48:], 1)
    assert np.isfinite(actual).all()


def test_mixed_masks_and_low_opacity_match_jax_and_d4():
    rng = np.random.default_rng(7)
    ink = rng.uniform(0, 0.03, (1, 35, 49, 1)).astype(np.float32)
    ink[:, 10:20, 18:31] *= 30
    source = 1 - 2 * ink
    noise = rng.choice([0.0, 0.0, 0.1, 0.5], source.shape).astype(np.float32)
    expected = runtime(source, noise)
    np.testing.assert_allclose(
        training(jnp.asarray(source), jnp.asarray(noise)), expected, atol=1e-7, rtol=1e-6
    )
    for k in range(4):
        for flip in (False, True):

            def transform(value, k=k, flip=flip):
                value = np.rot90(value, k, axes=(1, 2))
                return value[:, :, ::-1] if flip else value

            np.testing.assert_array_equal(
                runtime(transform(source), transform(noise)), transform(expected)
            )


def test_different_positive_strengths_remain_one_editable_region():
    rng = np.random.default_rng(8)
    source = rng.uniform(-1, 1, (1, 32, 64, 1)).astype(np.float32)
    noise = rng.uniform(0.001, 0.99, source.shape).astype(np.float32)
    np.testing.assert_array_equal(runtime(source, noise), normalized_guide(source))
