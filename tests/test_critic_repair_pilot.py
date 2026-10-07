import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from flax import nnx

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from train_critic_repair_pilot import candidate_gradient_penalty, critic_objective


class LinearCritic(nnx.Module):
    def __init__(self, weight):
        self.weight = nnx.Param(jnp.asarray(weight, jnp.float32))
        self.condition = nnx.Param(jnp.asarray(999.0, jnp.float32))

    def __call__(self, images):
        size = np.sqrt(images.shape[1] * images.shape[2])
        return self.weight * images[..., 2].sum((1, 2)) / size + self.condition * images[
            ..., 0
        ].sum((1, 2))


def test_gradient_penalty_excludes_condition_and_differentiates_critic_parameters():
    images = jnp.zeros((2, 4, 4, 3))
    alpha = jnp.full((2, 1, 1, 1), 0.3)
    model = LinearCritic(2)
    np.testing.assert_allclose(candidate_gradient_penalty(model, images, images, alpha), 1)
    loss, gradient = nnx.value_and_grad(
        lambda m: critic_objective(m, images, images, alpha, mode="wgan_gp")
    )(model)
    values = nnx.to_pure_dict(gradient)
    np.testing.assert_allclose(loss, 10)
    np.testing.assert_allclose(values["weight"], 20)
    np.testing.assert_allclose(values["condition"], 0)
    np.testing.assert_allclose(
        candidate_gradient_penalty(LinearCritic(1), images, images, alpha), 0, atol=1e-8
    )
