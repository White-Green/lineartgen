import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from lineartgen.training.feature_matching import feature_distance
from lineartgen.training.lineart_adversarial import PatchCritic


@pytest.mark.parametrize("metric", ["smooth_l1", "l2"])
def test_correct_gray_targets_are_stationary_and_protected_predictions_have_no_effect(metric):
    critic = PatchCritic(rngs=nnx.Rngs(3))
    target = jnp.linspace(-1, 1, 32 * 32).reshape(1, 32, 32, 1)
    condition = jnp.zeros((1, 32, 32, 2))
    active = jnp.ones_like(target, bool).at[:, :, 16:].set(False)
    def objective(p):
        return feature_distance(critic, p, target, condition, active, metric=metric).sum()
    value, gradient = jax.value_and_grad(objective)(target)
    assert float(value) == 0
    np.testing.assert_array_equal(gradient, 0)
    changed = target.at[:, :, 16:].set(-1)
    value, gradient = jax.value_and_grad(objective)(changed)
    assert float(value) == 0
    np.testing.assert_array_equal(gradient, 0)
    mixed = changed.at[:, 15, 15].set(1)
    value, gradient = jax.value_and_grad(objective)(mixed)
    assert float(value) > 0 and np.isfinite(gradient).all()
    np.testing.assert_array_equal(gradient[:, :, 16:], 0)
