import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from lineartgen.config import ExperimentConfig, LossConfig
from lineartgen.diffusion import (
    TrainingInputs,
    pyramid,
    recursive_forward,
    recursive_forward_with_preclip,
    sample_prepared,
)
from lineartgen.models import create_model
from lineartgen.models.output import denoised_output
from lineartgen.models.serialization import load_inference, save_inference
from lineartgen.training.interval import interval_squared_error
from lineartgen.training.lineart_steps import objective
from lineartgen.training.losses import image_loss
from lineartgen.training.steps import make_steps


def test_interval_error_equals_mse_inside_valid_range():
    z = jnp.linspace(-1, 1, 101)[:, None]
    y = jnp.linspace(-1, 1, 21)[None, :]
    np.testing.assert_array_equal(interval_squared_error(z, y), jnp.square(z - y))
    # Interior gray values have a singleton target, including nearly-white AA.
    y = jnp.array([0.999, -0.999, 0.0])
    np.testing.assert_allclose(
        interval_squared_error(jnp.array([2.0, -2.0, 3.0]), y),
        jnp.array([1.001**2, 1.001**2, 9.0]),
        rtol=1e-6,
    )


def test_endpoint_margins_stationary_wrong_saturation_receives_true_gradient():
    z = jnp.array([2.0, -2.0, 2.0, -2.0, 2.0, -2.0])
    y = jnp.array([1.0, -1.0, -1.0, 1.0, 0.8, -0.8])

    def f(x):
        return interval_squared_error(x, y).sum()

    expected = jnp.array([0.0, 0.0, 6.0, -6.0, 2.4, -2.4])
    np.testing.assert_allclose(jax.grad(f)(z), expected)
    eps = 1e-3
    for i in range(len(z)):
        e = jnp.zeros_like(z).at[i].set(eps)
        np.testing.assert_allclose((f(z + e) - f(z - e)) / (2 * eps), expected[i], atol=0.003)
    assert float(f(z - 0.1 * expected)) < float(f(z))
    np.testing.assert_array_equal(jax.grad(lambda x: interval_squared_error(x, y).sum())(y), 0)


def test_residual_preclip_is_physical_value_and_protected_gradient_stays_zero():
    image = jnp.linspace(-1, 1, 12).reshape(1, 3, 4, 1)
    noise = jnp.zeros_like(image).at[:, :, 2:].set(0.5)
    head = jnp.full_like(image, 4.0)
    clipped, preclip = denoised_output(head, image, noise, "residual", return_preclip=True)
    np.testing.assert_array_equal(preclip, image + jnp.sqrt(noise) * head)
    np.testing.assert_array_equal(clipped, jnp.clip(preclip, -1, 1))
    np.testing.assert_array_equal(preclip[:, :, :2], image[:, :, :2])
    config = LossConfig(pixel_loss="interval_mse", edge_weight=0, contrast_weight=0)

    def loss(h):
        prediction, raw = denoised_output(h, image, noise, "residual", return_preclip=True)
        return image_loss(prediction, image, noise, config, 0.001, preclip=raw)[0]

    gradient = jax.grad(loss)(head)
    np.testing.assert_array_equal(gradient[:, :, :2], 0)
    assert np.any(np.asarray(gradient[:, :, 2:]) > 0)
    with pytest.raises(ValueError, match="pre-clip"):
        image_loss(clipped, image, noise, config, 0.001)


def config_for_test(multiplier=0):
    return ExperimentConfig.model_validate(
        dict(
            model=dict(channels=[2, 4], convolutions_per_block=1, output_mode="residual"),
            loss=dict(pixel_loss="interval_mse", contrast_weight=0),
            diffusion=dict(scale_loss_multiplier=multiplier),
        )
    )


def test_recursive_feedback_is_clipped_not_preclip():
    class SaturatingModel:
        def __call__(self, image, noise, insert, *, return_preclip=False):
            # With two levels, only clipped feedback gives a top raw output of 3.
            raw = 2 + jax.image.resize(insert, image.shape, "nearest")
            clipped = jnp.clip(raw, -1, 1)
            return (clipped, raw) if return_preclip else clipped

    images = pyramid(jnp.zeros((1, 8, 8, 1)), 2)
    clipped, raw = recursive_forward_with_preclip(SaturatingModel(), images, images)
    np.testing.assert_array_equal(raw[0], 3)
    np.testing.assert_array_equal(raw[1], 2)
    jax.tree.map(
        np.testing.assert_array_equal, clipped, recursive_forward(SaturatingModel(), images, images)
    )


@pytest.mark.parametrize("multiplier", [0.0, 0.5])
def test_both_trainers_preserve_stage_weights_and_use_preclip(multiplier):
    config = config_for_test(multiplier)
    model = create_model(config.model, rngs=nnx.Rngs(7))
    # Force wrong saturation at both scales, making omitted preclip detectable.
    images = pyramid(jnp.full((1, 16, 16, 1), 3.0), 2)
    targets = pyramid(jnp.zeros_like(images[0]), 2)
    noise = pyramid(jnp.full_like(images[0], 0.5), 2)
    prepared = TrainingInputs(targets, noise, images)
    predictions, raw = recursive_forward_with_preclip(model, images, noise)
    expected = sum(
        multiplier**level
        * image_loss(p, y, n, config.loss, config.diffusion.min_noise, preclip=z)[0]
        for level, (p, z, y, n) in enumerate(zip(predictions, raw, targets, noise, strict=True))
    ) / (1 + multiplier)
    actual, _, outputs = objective(model, prepared, None, config)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    jax.tree.map(np.testing.assert_array_equal, outputs, recursive_forward(model, images, noise))
    np.testing.assert_allclose(make_steps(config)[2](model, prepared)["loss"], expected, rtol=5e-6)
    legacy = config.model_copy(
        update={"loss": config.loss.model_copy(update={"pixel_loss": "clipped_mse"})}
    )
    assert float(actual) > float(objective(model, prepared, None, legacy)[0])


def test_serialized_inference_and_export_keep_clipped_api(tmp_path):
    config = config_for_test()
    model = create_model(config.model, rngs=nnx.Rngs(8))
    image = jnp.linspace(-2, 2, 32 * 32).reshape(1, 32, 32, 1)
    noise = jnp.ones_like(image).at[:, :8].set(0)
    images, levels = pyramid(image, 3), pyramid(noise, 3)
    clipped, _ = recursive_forward_with_preclip(model, images, levels)
    jax.tree.map(np.testing.assert_array_equal, clipped, recursive_forward(model, images, levels))
    save_inference(model, config, tmp_path / "model", provenance="interval test")
    restored, saved = load_inference(tmp_path / "model")
    assert saved.loss.pixel_loss == "interval_mse"
    for steps in (1, 2, 5):
        np.testing.assert_array_equal(
            sample_prepared(model, images, levels, steps=steps),
            sample_prepared(restored, images, levels, steps=steps),
        )
    pytest.importorskip("jax2onnx")
    from lineartgen.export import export_model

    export_model(restored, saved, tmp_path / "model.onnx")


def test_reject_unsupported_surrogate_or_nested_combination():
    for model in [dict(output_clip_gradient="straight_through"), dict(architecture="nested")]:
        with pytest.raises(ValueError, match="single UNet with hard"):
            ExperimentConfig.model_validate(dict(model=model, loss=dict(pixel_loss="interval_mse")))
