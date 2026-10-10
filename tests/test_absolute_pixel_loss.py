import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lineartgen.config import LossConfig
from lineartgen.training.losses import absolute_error, image_loss


def test_absolute_error_value_direction_and_zero_gradient_at_exact_match():
    error = jnp.asarray([-2.0, -0.01, 0.0, 0.01, 2.0])
    value, grad = jax.value_and_grad(lambda x: absolute_error(x, 0).sum())(error)
    np.testing.assert_allclose(value, 4.02)
    np.testing.assert_array_equal(grad, [-1, -1, 0, 1, 1])


@pytest.mark.parametrize("floor", [0.05, 1.0])
def test_l1_pixel_loss_mask_and_strength_weight_are_preserved(floor):
    target = jnp.asarray([[[[-1.0], [0.25], [1.0], [0.5]]]])
    pred = jnp.asarray([[[[-0.5], [0.25], [0.75], [-0.5]]]])
    level = jnp.asarray([[[[0.1], [0.1], [0.5], [0.0]]]])
    config = LossConfig(
        pixel_loss="clipped_l1",
        noise_weight_floor=floor,
        balance_tones=False,
        edge_weight=0,
        identity_weight=0,
        contrast_weight=0,
    )
    value, grad = jax.value_and_grad(lambda x: image_loss(x, target, level, config, 0.001)[0])(pred)
    w1, w2 = max(0.1, floor), max(0.5, floor)
    np.testing.assert_allclose(value, (0.5 / w1 + 0.25 / w2) / 3, rtol=1e-6)
    np.testing.assert_allclose(grad.ravel(), [1 / (3 * w1), 0, -1 / (3 * w2), 0], rtol=1e-6)
    assert float(image_loss(pred, target, level, config, 0.001)[1]["identity"]) == 1.0


def test_l1_does_not_push_correct_antialias_values_to_black_or_white():
    target = jnp.linspace(-1, 1, 9).reshape(1, 3, 3, 1)
    level = jnp.ones_like(target)
    config = LossConfig(pixel_loss="clipped_l1", edge_weight=0, contrast_weight=0)
    value, grad = jax.value_and_grad(lambda x: image_loss(x, target, level, config, 0.001)[0])(
        target
    )
    assert float(value) == 0
    np.testing.assert_array_equal(grad, np.zeros_like(target))


def test_mse_default_unchanged_and_loss_names_validated():
    target = jnp.zeros((1, 2, 2, 1))
    pred = jnp.full_like(target, 0.25)
    noise = jnp.full_like(target, 0.5)
    assert LossConfig().pixel_loss == "clipped_mse"
    np.testing.assert_allclose(
        image_loss(pred, target, noise, LossConfig(), 0.001)[1]["denoise"], 0.125
    )
    with pytest.raises(ValueError):
        LossConfig(pixel_loss="l1_typo")
