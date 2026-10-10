import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lineartgen.config import LossConfig
from lineartgen.training.losses import image_loss, tone_weights


@pytest.mark.parametrize("power", [0.0, 0.5, 1.0])
def test_disabled_local_balance_matches_frozen_R_exactly(power):
    path = Path("../jax-source-interval-l1-20261010/src/lineartgen/training/losses.py").resolve()
    spec = importlib.util.spec_from_file_location("lineartgen.training.frozen_R_losses", path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    image = jax.random.uniform(jax.random.PRNGKey(1), (2, 32, 32, 1), minval=-1, maxval=1)
    np.testing.assert_array_equal(tone_weights(image, power), legacy.tone_weights(image, power))


@pytest.mark.parametrize("gray", [-1.0, 0.0, 0.4, 1.0])
def test_constant_fields_have_finite_uniform_weights_even_at_boundaries(gray):
    image = jnp.full((2, 32, 32, 1), gray)
    weights = np.asarray(tone_weights(image, 0.5, 8))
    assert np.isfinite(weights).all()
    np.testing.assert_allclose(weights, 1, atol=2e-6, rtol=0)


def test_soft_target_weights_are_complement_symmetric_and_normalized():
    image = jax.random.uniform(jax.random.PRNGKey(2), (2, 32, 32, 1), minval=-1, maxval=1)
    weights = tone_weights(image, 0.5, 8)
    np.testing.assert_allclose(weights.mean((1, 2)), 1, atol=2e-6, rtol=0)
    np.testing.assert_allclose(weights, tone_weights(-image, 0.5, 8), atol=2e-6, rtol=0)


@pytest.mark.parametrize("complement", [False, True])
def test_narrow_minority_ink_and_paper_get_more_relative_weight(complement):
    image = jnp.ones((1, 64, 64, 1)).at[:, 15:49, 10:22].set(-1).at[:, 15:49, 45].set(-1)
    if complement:
        image = -image
    old, new = [np.asarray(tone_weights(image, 0.5, radius)) for radius in (0, 8)]
    old_ratio = old[0, 32, 45, 0] / old[0, 32, 16, 0]
    new_ratio = new[0, 32, 45, 0] / new[0, 32, 16, 0]
    assert new_ratio > old_ratio * 3


def test_interval_gradients_repair_faint_ink_and_false_bridges_and_keep_soft_teacher():
    alpha = jnp.zeros((1, 96, 96, 1)).at[:, 15:80, 5:35].set(1)
    alpha = alpha.at[:, 15:80, 20].set(0).at[:, 15:80, 70].set(0.8)
    target = 1 - 2 * alpha
    damaged = alpha.at[:, 15:80, 20].set(0.4).at[:, 15:80, 70].set(0.25)
    prediction = 1 - 2 * damaged
    levels = jnp.full_like(target, 0.5)
    configs = [
        LossConfig(
            pixel_loss="interval_l1",
            tone_balance_power=0.5,
            local_tone_radius=r,
            edge_weight=0,
            contrast_weight=0,
            identity_weight=0,
        )
        for r in (0, 8)
    ]
    grads = []
    for config in configs:

        def loss(p, config=config):
            return image_loss(jnp.clip(p, -1, 1), target, levels, config, 0.001, preclip=p)[0]

        grad = np.asarray(jax.grad(loss)(prediction))
        assert np.isfinite(grad).all() and grad[0, 40, 20, 0] < 0 and grad[0, 40, 70, 0] > 0
        assert float(loss(target)) == 0
        np.testing.assert_array_equal(jax.grad(loss)(target), 0)
        grads.append(grad)
    assert abs(grads[1][0, 40, 20, 0]) > abs(grads[0][0, 40, 20, 0])
    assert abs(grads[1][0, 40, 70, 0]) > abs(grads[0][0, 40, 70, 0])


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(local_tone_radius=-1),
        dict(local_tone_radius=17),
        dict(local_tone_radius=8, balance_tones=False),
        dict(local_tone_radius=8, tone_balance_power=0),
    ],
)
def test_invalid_local_balance_configs(kwargs):
    with pytest.raises(ValueError):
        LossConfig(**kwargs)
