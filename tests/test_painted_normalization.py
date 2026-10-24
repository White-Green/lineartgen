import jax.numpy as jnp
import numpy as np
import pytest
from lineartgen.models.painted_normalization import normalized_painted_guide as training
from lineartgen_runtime.graph import normalized_guide
from lineartgen_runtime.masked_normalization import normalized_masked_guide
from lineartgen_runtime.painted_normalization import normalized_painted_guide as runtime


@pytest.mark.parametrize("strength", [0.0, 0.1, 0.5, 0.75])
def test_uniform_masks_keep_the_original_guide(strength):
    source = np.random.default_rng(1).uniform(-1, 1, (1, 35, 49, 1)).astype(np.float32)
    noise = np.full_like(source, strength)
    np.testing.assert_array_equal(runtime(source, noise), normalized_guide(source))


def test_binary_training_masks_are_exactly_equivalent_to_protected_policy():
    rng = np.random.default_rng(3)
    source = rng.uniform(-1, 1, (4, 35, 49, 1)).astype(np.float32)
    noise = (
        rng.integers(2, size=source.shape).astype(np.float32)
        * np.array([0.1, 0.25, 0.5, 0.75], np.float32)[:, None, None, None]
    )
    np.testing.assert_array_equal(runtime(source, noise), normalized_masked_guide(source, noise))


def test_antialiased_painted_edges_do_not_attenuate_the_continuing_sketch():
    ink = np.zeros((1, 64, 96, 1), np.float32)
    noise = np.full_like(ink, 0.5)
    ink[:, 29:35, :48] = 1
    noise[:, 29:35, :48] = 0
    for y in (28, 35):
        ink[:, y, :48] = 0.6
        noise[:, y, :48] = 0.25
    ink[:, 29:35, 48:] = 0.4
    source = 1 - 2 * ink
    previous = normalized_masked_guide(source, noise)
    actual = runtime(source, noise)
    assert previous[0, 32, 48, 0] > 0.4
    assert actual[0, 32, 48, 0] == pytest.approx(0.2)
    assert actual[0, 28, 40, 0] == pytest.approx(0.52)


def test_soft_masks_match_jax_without_batch_crosstalk_or_strength_dependence():
    rng = np.random.default_rng(9)
    source = rng.uniform(-1, 1, (3, 35, 49, 1)).astype(np.float32)
    alpha = rng.integers(0, 256, size=source.shape).astype(np.float32) / 255
    alpha[:, :, ::4] = 0
    noise = (1 - alpha) * np.array([0.1, 0.5, 0.75], np.float32)[:, None, None, None]
    expected = runtime(source, noise)
    np.testing.assert_allclose(
        training(jnp.asarray(source), jnp.asarray(noise)), expected, atol=1e-7, rtol=1e-6
    )
    for index in range(3):
        np.testing.assert_array_equal(
            runtime(source[index : index + 1], noise[index : index + 1]),
            expected[index : index + 1],
        )
    np.testing.assert_array_equal(runtime(source, noise * 0.5), expected)
