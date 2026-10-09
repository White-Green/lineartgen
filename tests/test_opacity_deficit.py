import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lineartgen.config import LossConfig
from lineartgen.lineart_metrics import halo_metrics
from lineartgen.training.losses import image_loss
from lineartgen.training.opacity import opacity_deficit_loss


def test_matches_numpy_quality_metric_with_partial_and_empty_ink():
    rng = np.random.default_rng(91)
    target = rng.uniform(-1, 1, (4, 9, 11, 1)).astype(np.float32)
    target[2] = 1  # Blank teacher.
    prediction = rng.uniform(-1.2, 1.2, target.shape).astype(np.float32)
    active = rng.random(target.shape) > 0.3
    active[3] = False  # Fully protected.
    expected = [
        halo_metrics(p, y, a.astype(np.float32))["ink_deficit_fraction"] or 0.0
        for p, y, a in zip(prediction, target, active, strict=True)
    ]
    loss = jax.jit(opacity_deficit_loss)(prediction, target, active)
    np.testing.assert_allclose(loss, np.mean(expected), rtol=2e-6)


def test_gradient_retains_weak_ink_without_darkening_correct_or_excess_ink():
    # Missing dark/gray ink, correct gray, excess ink, background, protected ink.
    truth = jnp.array([0.9, 0.1, 0.3, 0.3, 0.0, 0.9]).reshape(1, 1, 6, 1)
    ink = jnp.array([0.2, 0.05, 0.3, 0.5, 0.2, 0.1]).reshape(truth.shape)
    active = jnp.array([True, True, True, True, True, False]).reshape(truth.shape)
    target, prediction = 1 - 2 * truth, 1 - 2 * ink
    grad = jax.grad(opacity_deficit_loss)(prediction, target, active)
    assert float(grad[0, 0, 0, 0]) > 0  # Gradient descent lowers white-valued prediction.
    np.testing.assert_array_equal(grad[..., 0:1, :], grad[..., 1:2, :])
    np.testing.assert_array_equal(grad[..., 2:, :], 0)
    np.testing.assert_array_equal(jax.grad(opacity_deficit_loss, 1)(prediction, target, active), 0)


def test_linear_force_remains_constant_as_deficit_shrinks():
    target = jnp.full((1, 4, 4, 1), -0.6)
    active = jnp.ones_like(target, dtype=bool)
    gradient = jax.grad(opacity_deficit_loss)
    np.testing.assert_array_equal(
        gradient(target + 0.001, target, active), gradient(target + 0.9, target, active)
    )
    assert float(opacity_deficit_loss(target + 0.001, target, active)) > 0
    assert float(opacity_deficit_loss(target, target, active)) == 0
    np.testing.assert_array_equal(gradient(target, target, active), 0)


def test_blank_and_protected_losses_and_gradients_are_finite_zero():
    target = jnp.ones((2, 4, 4, 1)).at[1].set(-0.8)
    prediction = jnp.zeros_like(target)
    active = jnp.ones_like(target, dtype=bool).at[1].set(False)
    loss, grad = jax.jit(jax.value_and_grad(opacity_deficit_loss))(prediction, target, active)
    assert float(loss) == 0
    np.testing.assert_array_equal(grad, 0)


def test_image_loss_coefficient_and_min_noise_mask():
    target = jnp.full((1, 4, 4, 1), -0.2)
    prediction = target + 0.1
    noise = jnp.full_like(target, 0.1).at[:, :2].set(0.001)
    config = LossConfig(tone_balance_power=0.5)
    with_deficit = config.model_copy(update={"opacity_deficit_weight": 0.3})
    before, parts = image_loss(prediction, target, noise, config, 0.001)
    after, updated = image_loss(prediction, target, noise, with_deficit, 0.001)
    assert "opacity_deficit" not in parts
    expected = opacity_deficit_loss(prediction, target, noise > 0.001)
    np.testing.assert_allclose(after - before, 0.3 * expected, rtol=2e-6)
    np.testing.assert_array_equal(updated["opacity_deficit"], expected)

    def loss(p, c):
        return image_loss(p, target, noise, c, 0.001)[0]

    gradient_change = jax.grad(loss)(prediction, with_deficit) - jax.grad(loss)(prediction, config)
    np.testing.assert_array_equal(gradient_change[:, :2], 0)
    assert np.all(gradient_change[:, 2:] > 0)


def test_coefficient_rejects_negative_and_nonfinite_values():
    assert LossConfig().opacity_deficit_weight == 0
    for value in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            LossConfig(opacity_deficit_weight=value)
