import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lineartgen.config import LossConfig
from lineartgen.training.losses import image_loss


def test_uniform_floor_removes_strength_weight_from_pixel_loss_and_gradient():
    target = jnp.full((2, 16, 16, 1), -0.5)
    prediction = jnp.zeros_like(target)
    gradients = []
    for strength in (0.1, 0.5):
        noise = jnp.full_like(target, strength)
        config = LossConfig(noise_weight_floor=1)

        def denoise(x, noise=noise, config=config):
            return image_loss(x, target, noise, config, 0.001)[1]["denoise"]

        value, grad = jax.value_and_grad(denoise)(prediction)
        np.testing.assert_allclose(value, 0.25)
        gradients.append(grad)
        inverse = image_loss(
            prediction,
            target,
            noise,
            config.model_copy(update={"noise_weight_floor": 0.05}),
            0.001,
        )[1]["denoise"]
        np.testing.assert_allclose(inverse, 0.25 / strength, rtol=1e-6)
    np.testing.assert_array_equal(*gradients)
    with pytest.raises(ValueError):
        LossConfig(noise_weight_floor=1.01)
