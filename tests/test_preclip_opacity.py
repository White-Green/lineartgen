import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from lineartgen.config import ExperimentConfig, LossConfig
from lineartgen.diffusion import TrainingInputs, pyramid, recursive_forward
from lineartgen.models import create_model
from lineartgen.training.lineart_steps import objective
from lineartgen.training.losses import image_loss
from lineartgen.training.opacity import opacity_deficit_loss
from lineartgen.training.steps import make_steps


def test_physical_range_agreement_and_white_saturation_gradient():
    target = jnp.array([-0.8, 0.6, 0.2, 0.9]).reshape(1, 1, 4, 1)
    active = jnp.ones_like(target, dtype=bool)

    def extended(x):
        return opacity_deficit_loss(x, target, active, unclipped=True)

    def clipped(x):
        return opacity_deficit_loss(jnp.clip(x, -1, 1), target, active)

    raw = target + 0.05
    np.testing.assert_array_equal(extended(raw), clipped(raw))
    np.testing.assert_array_equal(jax.grad(extended)(raw), jax.grad(clipped)(raw))
    raw = jnp.full_like(target, 1.3)
    assert np.all(jax.grad(extended)(raw) > 0)
    np.testing.assert_array_equal(jax.grad(clipped)(raw), 0)
    direction = jax.grad(extended)(raw)
    eps = 1e-3
    numerical = (extended(raw + eps) - extended(raw - eps)) / (2 * eps)
    np.testing.assert_allclose(numerical, direction.sum(), rtol=2e-4)


def test_correct_gray_dark_excess_paper_and_protected_have_no_extra_force():
    target = jnp.array([0.2, -0.8, 1.0, -0.5]).reshape(1, 1, 4, 1)
    raw = jnp.array([0.2, -2.0, 1.5, 1.5]).reshape(target.shape)
    active = jnp.array([True, True, True, False]).reshape(target.shape)
    def fn(p):
        return opacity_deficit_loss(p, target, active, unclipped=True)

    assert float(fn(raw)) == 0
    np.testing.assert_array_equal(jax.grad(fn)(raw), 0)


def test_image_loss_requires_and_uses_physical_preclip():
    target = jnp.zeros((1, 4, 4, 1))
    raw = jnp.full_like(target, 1.4)
    noise = jnp.full_like(target, 0.5).at[:, :2].set(0)
    config = LossConfig(opacity_deficit_weight=0.3, opacity_deficit_preclip=True)
    with pytest.raises(ValueError, match="physical pre-clip"):
        image_loss(jnp.clip(raw, -1, 1), target, noise, config, 0.001)

    def loss(p):
        return image_loss(jnp.clip(p, -1, 1), target, noise, config, 0.001, preclip=p)[0]

    grad = jax.grad(loss)(raw)
    np.testing.assert_array_equal(grad[:, :2], 0)
    assert np.all(grad[:, 2:] > 0)


@pytest.mark.parametrize("multiplier", [0.0, 0.5])
def test_both_training_paths_reach_parameters_and_preserve_clipped_feedback(multiplier):
    config = ExperimentConfig.model_validate(
        dict(
            model=dict(channels=[2, 4], convolutions_per_block=1, output_mode="residual"),
            loss=dict(
                opacity_deficit_weight=0.3,
                opacity_deficit_preclip=True,
                edge_weight=0,
                contrast_weight=0,
            ),
            diffusion=dict(scale_loss_multiplier=multiplier),
        )
    )
    model = create_model(config.model, rngs=nnx.Rngs(7))
    images = pyramid(jnp.full((1, 16, 16, 1), 3.0), 2)
    noise = pyramid(jnp.full_like(images[0], 0.5), 2)
    targets = pyramid(jnp.zeros_like(images[0]), 2)
    prepared = TrainingInputs(targets, noise, images)
    value, _, outputs = objective(model, prepared, None, config)
    jax.tree.map(np.testing.assert_array_equal, outputs, recursive_forward(model, images, noise))
    gradients, _, evaluate = make_steps(config)
    np.testing.assert_allclose(evaluate(model, prepared)["loss"], value, rtol=5e-6)
    metrics, grads = gradients(model, prepared)
    assert np.isfinite(float(metrics["loss"]))
    assert any(np.any(np.asarray(x) != 0) for x in jax.tree.leaves(grads))
    (_, direct_grads) = nnx.value_and_grad(lambda m: objective(m, prepared, None, config)[0])(model)
    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6), grads, direct_grads
    )
    disabled = config.model_copy(
        update={"loss": config.loss.model_copy(update={"opacity_deficit_preclip": False})}
    )
    old_grads = make_steps(disabled)[0](model, prepared)[1]
    assert float(value) > float(objective(model, prepared, None, disabled)[0])
    assert all(np.all(np.asarray(x) == 0) for x in jax.tree.leaves(old_grads))


def test_invalid_or_unsupported_preclip_configs_are_rejected():
    with pytest.raises(ValueError, match="positive opacity"):
        LossConfig(opacity_deficit_preclip=True)
    for model in [dict(architecture="nested"), dict(output_clip_gradient="straight_through")]:
        with pytest.raises(ValueError, match="single UNet or local CNN with hard"):
            ExperimentConfig.model_validate(
                dict(
                    model=model,
                    loss=dict(
                        opacity_deficit_weight=0.3,
                        opacity_deficit_preclip=True,
                    ),
                )
            )
