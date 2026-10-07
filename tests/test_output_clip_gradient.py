import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from lineartgen.config import ExperimentConfig, ModelConfig
from lineartgen.diffusion import pyramid, recursive_forward, sample_prepared
from lineartgen.models import create_model
from lineartgen.models.output import denoised_output, straight_through_clip
from lineartgen.models.serialization import load_inference, save_inference


def test_wrong_saturated_values_receive_inward_surrogate_gradients():
    values = jnp.asarray([-9.0, -1.2, -1.0, -0.4, 0.0, 0.4, 1.0, 1.2, 9.0])
    teacher = jnp.zeros_like(values)

    def hard(x):
        return jnp.square(jnp.clip(x, -1, 1) - teacher).sum()

    def surrogate(x):
        return jnp.square(straight_through_clip(x) - teacher).sum()

    np.testing.assert_array_equal(straight_through_clip(values), jnp.clip(values, -1, 1))
    actual = jax.grad(surrogate)(values)
    np.testing.assert_array_equal(actual, 2 * jnp.clip(values, -1, 1))
    np.testing.assert_array_equal(np.asarray(jax.grad(hard)(values))[[0, 1, 7, 8]], 0)
    assert float(surrogate(values - 0.2 * actual)) < float(surrogate(values))

    def correct(x):
        return jnp.square(straight_through_clip(x) - 1).sum()

    np.testing.assert_array_equal(jax.grad(correct)(jnp.asarray([2.0, 9.0])), 0)


def test_residual_identity_and_protected_head_gradient_are_exact():
    image = jnp.linspace(-1, 1, 12).reshape(1, 3, 4, 1)
    raw = jnp.full_like(image, 4.0)
    noise = jnp.zeros_like(image).at[:, :, 2:, :].set(0.5)

    def call(x):
        return denoised_output(x, image, noise, "residual", "straight_through")

    np.testing.assert_array_equal(call(raw)[:, :, :2], image[:, :, :2])
    gradient = jax.grad(lambda x: call(x).sum())(raw)
    np.testing.assert_array_equal(gradient[:, :, :2], 0)
    np.testing.assert_allclose(gradient[:, :, 2:], np.sqrt(0.5))


def test_recursive_forward_and_serialized_inference_are_bitwise_identical(tmp_path):
    config = ExperimentConfig.model_validate(
        dict(model=dict(channels=[2, 4], convolutions_per_block=1, output_mode="residual"))
    )
    hard = create_model(config.model, rngs=nnx.Rngs(5))
    soft = nnx.clone(hard)
    soft.output_clip_gradient = "straight_through"
    rng = np.random.default_rng(7)
    image = jnp.asarray(rng.uniform(-2, 2, (1, 32, 32, 1)), dtype=jnp.float32)
    noise = jnp.ones_like(image).at[:, :8, :].set(0)
    images, levels = pyramid(image, 3), pyramid(noise, 3)
    for steps in (1, 2, 5):
        np.testing.assert_array_equal(
            sample_prepared(hard, images, levels, steps=steps),
            sample_prepared(soft, images, levels, steps=steps),
        )
    config = config.model_copy(
        update={
            "model": config.model.model_copy(update={"output_clip_gradient": "straight_through"})
        }
    )
    save_inference(soft, config, tmp_path / "model", provenance="unit test")
    restored, saved = load_inference(tmp_path / "model")
    assert saved.model.output_clip_gradient == "straight_through"
    jax.tree.map(
        np.testing.assert_array_equal,
        recursive_forward(hard, images, levels),
        recursive_forward(restored, images, levels),
    )
    with pytest.raises(ValueError, match="require architecture"):
        ModelConfig(architecture="nested", output_clip_gradient="straight_through")


def test_surrogate_model_exports_as_an_ordinary_inference_graph(tmp_path):
    pytest.importorskip("jax2onnx")
    from lineartgen.export import export_model

    config = ExperimentConfig.model_validate(
        dict(
            model=dict(
                channels=[2, 4],
                convolutions_per_block=1,
                output_mode="residual",
                output_clip_gradient="straight_through",
            )
        )
    )
    model = create_model(config.model, rngs=nnx.Rngs(8))
    export_model(model, config, tmp_path / "model.onnx")
    assert model.output_clip_gradient == "straight_through"
    assert (tmp_path / "model.onnx").exists()
