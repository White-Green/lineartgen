import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from fill_hard_negative import hard_negative_bce
from learned_centerline_model import balanced_bce


@pytest.mark.parametrize("label", [0.0, 1.0])
def test_uniform_targets_finite_with_correct_sign(label):
    x = jnp.arange(8.0).reshape(2, 2, 2) - 4
    y = jnp.full_like(x, label)
    loss, gradient = jax.jit(jax.value_and_grad(hard_negative_bce))(x, y)
    assert np.isfinite(loss) and np.isfinite(gradient).all()
    assert (np.asarray(gradient) >= 0).all() if label == 0 else (gradient <= 0).all()
    expected = optax.sigmoid_binary_cross_entropy(x, y).reshape(2, -1)
    expected = expected.max(-1).mean() if label == 0 else expected.mean()
    np.testing.assert_allclose(loss, expected, rtol=1e-6)


def test_positive_cells_cannot_be_selected_as_negatives():
    x = jnp.array([[[-100.0, 2.0, 1.0, -3.0]]])
    y = jnp.array([[[1.0, 0.0, 0.0, 0.0]]])
    value, grad = jax.value_and_grad(hard_negative_bce)(x, y)
    np.testing.assert_allclose(value, 100 + jax.nn.softplus(2.0), rtol=1e-6)
    np.testing.assert_allclose(grad, [[[-1.0, jax.nn.sigmoid(2.0), 0.0, 0.0]]], rtol=1e-6)


def test_fewer_negatives_than_k_use_actual_count():
    x = jnp.array([[[0.0, -1.0, -2.0, 3.0]]])
    y = jnp.array([[[1.0, 1.0, 1.0, 0.0]]])
    np.testing.assert_allclose(hard_negative_bce(x, y, 0.5), balanced_bce(x, y), rtol=1e-6)


def test_full_tail_matches_current_loss_and_gradient_per_image():
    x = jnp.linspace(-5, 5, 24).reshape(2, 3, 4)
    y = (jnp.arange(24).reshape(2, 3, 4) % 3 == 0).astype(jnp.float32)
    a = jax.value_and_grad(lambda v: hard_negative_bce(v, y, 1.0))(x)
    b = jax.value_and_grad(lambda v: balanced_bce(v, y))(x)
    for aa, bb in zip(a, b, strict=True):
        np.testing.assert_allclose(aa, bb, atol=1e-7, rtol=1e-6)


def test_invalid_fraction():
    with pytest.raises(ValueError):
        hard_negative_bce(jnp.zeros((1, 2)), jnp.zeros((1, 2)), 0)
