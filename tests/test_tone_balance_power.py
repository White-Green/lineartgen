import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lineartgen.config import LossConfig
from lineartgen.training.losses import image_loss, tone_weights


def test_default_preserves_existing_weights_and_zero_is_unit():
    clean = jnp.linspace(-1, 1, 2048).reshape(2, 32, 32, 1)
    black = jnp.clip((1 - clean) * 0.5, 0, 1)
    white = 1 - black
    w = black / (2 * black.mean((1, 2), keepdims=True) + 1e-6)
    w += white / (2 * white.mean((1, 2), keepdims=True) + 1e-6)
    old = w / (w.mean((1, 2), keepdims=True) + 1e-6)
    np.testing.assert_array_equal(tone_weights(clean), old)
    np.testing.assert_array_equal(tone_weights(clean, 0), jnp.ones_like(clean))


def test_tempering_reduces_rare_ink_ratio_and_keeps_blank_finite():
    clean = jnp.ones((3, 64, 64, 1)).at[0, :, 31:33].set(-1).at[2].set(-1)
    full, tempered = tone_weights(clean), tone_weights(clean, 0.5)
    ratio = full[0, 32, 31, 0] / full[0, 32, 0, 0]
    tempered_ratio = tempered[0, 32, 31, 0] / tempered[0, 32, 0, 0]
    np.testing.assert_allclose(tempered_ratio, jnp.sqrt(ratio), rtol=1e-6)
    assert 1 < tempered_ratio < ratio
    assert np.isfinite(tempered).all()
    np.testing.assert_allclose(tempered.mean((1, 2)), 1, atol=2e-6)


def test_unit_power_matches_unbalanced_loss_and_gradient():
    clean = jnp.linspace(-1, 1, 1024).reshape(1, 32, 32, 1)
    prediction = jnp.sin(clean)
    noise = jnp.full_like(clean, 0.1).at[:, :8].set(0)

    def loss(p, config):
        return image_loss(p, clean, noise, config, 0.001)[0]

    a = LossConfig(tone_balance_power=0)
    b = LossConfig(balance_tones=False)
    np.testing.assert_array_equal(loss(prediction, a), loss(prediction, b))
    np.testing.assert_array_equal(jax.grad(loss)(prediction, a), jax.grad(loss)(prediction, b))


def test_tempered_weighted_optimum_has_zero_actual_pixel_gradient():
    teachers = jnp.ones((3, 16, 16, 1))
    for i in range(3):
        teachers = teachers.at[i, :, 6 + i : 8 + i].set(-1)
    prior = jnp.asarray([0.2, 0.5, 0.3]).reshape(3, 1, 1, 1)
    weights = tone_weights(teachers, 0.5)
    optimum = (prior * weights * teachers).sum(0, keepdims=True) / (prior * weights).sum(
        0, keepdims=True
    )
    config = LossConfig(tone_balance_power=0.5, edge_weight=0, contrast_weight=0)

    def risk(prediction):
        return sum(
            prior[i, 0, 0, 0]
            * image_loss(
                prediction, teachers[i : i + 1], jnp.full_like(prediction, 0.1), config, 0.001
            )[0]
            for i in range(3)
        )

    assert np.max(np.abs(jax.grad(risk)(optimum))) < 2e-8
    assert np.isfinite(float(risk(optimum)))


def test_power_range_is_validated():
    for value in (-0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            LossConfig(tone_balance_power=value)
