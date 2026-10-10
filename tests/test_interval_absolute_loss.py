import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lineartgen.config import ExperimentConfig, LossConfig
from lineartgen.training.interval import interval_absolute_error
from lineartgen.training.losses import absolute_error, image_loss


def test_interval_l1_has_inward_wrong_saturation_gradients_and_valid_margins():
    prediction = jnp.array([-2.0, -2.0, -0.2, 0.3, 2.0, 2.0, 3.0])
    target = jnp.array([1.0, -1.0, -0.2, 0.3, 1.0, -1.0, 0.25])
    np.testing.assert_array_equal(
        interval_absolute_error(prediction, target), [3, 0, 0, 0, 0, 3, 2.75]
    )
    gradient = jax.grad(lambda x: interval_absolute_error(x, target).sum())(prediction)
    np.testing.assert_array_equal(gradient, [-1, 0, 0, 0, 0, 1, 1])
    np.testing.assert_array_equal(
        jax.grad(lambda x: interval_absolute_error(x, target).sum())(target), 0
    )


def test_interval_l1_matches_clipped_l1_in_range_including_antialias_values():
    prediction = jnp.linspace(-1, 1, 13)
    target = jnp.roll(prediction, 3)
    np.testing.assert_array_equal(
        interval_absolute_error(prediction, target), absolute_error(prediction, target)
    )


@pytest.mark.parametrize("floor", [0.05, 1.0])
def test_interval_l1_real_loss_preserves_masks_and_strength_scaling(floor):
    config = LossConfig(
        pixel_loss="interval_l1",
        balance_tones=False,
        edge_weight=0,
        contrast_weight=0,
        identity_weight=0.1,
        noise_weight_floor=floor,
    )
    physical = jnp.array([-2.0, 2.0, -2.0, 2.0]).reshape(1, 2, 2, 1)
    target = jnp.array([1.0, -1.0, 1.0, -1.0]).reshape(1, 2, 2, 1)
    strength = jnp.array([0.5, 0.5, 0.0, 0.0]).reshape(1, 2, 2, 1)

    def loss(z):
        return image_loss(jnp.clip(z, -1, 1), target, strength, config, 0.0, preclip=z)[0]

    scale = max(0.5, floor)
    np.testing.assert_allclose(loss(physical), 3 / scale + 0.3, rtol=1e-6)
    np.testing.assert_allclose(
        jax.grad(loss)(physical).ravel(), [-0.5 / scale, 0.5 / scale, -0.05, 0.05], rtol=1e-6
    )
    with pytest.raises(ValueError, match="interval_l1 requires"):
        image_loss(jnp.clip(physical, -1, 1), target, strength, config, 0.0)


def test_interval_l1_rejects_surrogate_clip_gradient():
    with pytest.raises(ValueError, match="interval_l1 requires"):
        ExperimentConfig.model_validate(
            dict(
                model=dict(output_clip_gradient="straight_through"),
                loss=dict(pixel_loss="interval_l1"),
            )
        )
    assert LossConfig().pixel_loss == "clipped_mse"
