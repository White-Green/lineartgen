import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from lineartgen_runtime import LineartModel
from lineartgen_runtime.images import prepare_input, pyramid_plan, to_bgra

from lineartgen.config import ExperimentConfig, ModelConfig
from lineartgen.diffusion import pyramid, recursive_forward, sample_prepared
from lineartgen.evaluation_sampling import inference_with_source, runtime_inputs
from lineartgen.export import export_model
from lineartgen.models import create_model
from lineartgen.models.serialization import load_inference, save_inference
from lineartgen.training.lineart_study import assert_initial_parameters, create_arm


def configuration(width=0):
    return ExperimentConfig.model_validate(
        dict(
            model=dict(
                channels=[2, 4],
                convolutions_per_block=2,
                source_conditioning=True,
                output_mode="residual",
                detail_channels=width,
            )
        )
    )


@pytest.mark.parametrize("width,extra", [(8, 1313), (16, 3777)])
def test_zero_detail_preserves_inherited_function_and_serializes(width, extra, tmp_path):
    c = configuration()
    base = create_model(c.model, rngs=nnx.Rngs(42))
    changed_config = configuration(width)
    model = create_arm(changed_config, base)["model"]
    assert_initial_parameters(base, model)
    # Construction after inherited layers must also preserve their random initialization.
    assert_initial_parameters(base, create_model(changed_config.model, rngs=nnx.Rngs(42)))
    source = np.random.default_rng(10).uniform(-1, 1, (1, 32, 32, 1)).astype(np.float32)
    level = np.full_like(source, 0.5)
    level[:, :8] = 0
    inputs = runtime_inputs(source, level, 23, 3)
    guides = pyramid(jnp.asarray(source), 3)
    jax.tree.map(
        np.testing.assert_array_equal,
        recursive_forward(base, *inputs, guides=guides),
        recursive_forward(model, *inputs, guides=guides),
    )
    for steps in (1, 2, 5):
        np.testing.assert_array_equal(
            sample_prepared(base, *inputs, steps=steps, guides=guides),
            sample_prepared(model, *inputs, steps=steps, guides=guides),
        )
    save_inference(model, changed_config, tmp_path / "model", provenance="zero detail fixture")
    restored, saved = load_inference(tmp_path / "model")
    assert saved == changed_config
    jax.tree.map(np.testing.assert_array_equal, nnx.state(model), nnx.state(restored))
    fullbase = create_model(ModelConfig(source_conditioning=True), rngs=nnx.Rngs(42))
    full = create_model(
        ModelConfig(source_conditioning=True, detail_channels=width), rngs=nnx.Rngs(42)
    )

    def count(m):
        return sum(x.size for x in jax.tree.leaves(nnx.state(m, nnx.Param)))

    assert count(full) - count(fullbase) == extra


def test_detail_projection_then_hidden_layers_receive_gradient_and_zero_mask_is_exact():
    model = create_model(configuration(8).model, rngs=nnx.Rngs(42))
    image = jnp.linspace(-0.4, 0.4, 16 * 16).reshape(1, 16, 16, 1)
    level = jnp.full_like(image, 0.3).at[:, :4].set(0)
    insert = jnp.zeros((1, 8, 8, 1))
    guide = jnp.full_like(image, 0.6)

    def loss(m):
        return jnp.mean((m(image, level, insert, guide) - 0.2) ** 2)

    gradient = nnx.grad(loss)(model)
    last = gradient["detail_head"]["layers"][2]
    assert np.linalg.norm(np.asarray(last["kernel"])) > 0
    # One ordinary gradient step opens the path through the last projection.
    state = nnx.state(model, nnx.Param)
    nnx.update(model, jax.tree.map(lambda p, g: p - 0.1 * g, state, gradient))
    gradient = nnx.grad(loss)(model)
    for i in range(3):
        assert np.linalg.norm(np.asarray(gradient["detail_head"]["layers"][i]["kernel"])) > 0
    out = np.asarray(model(image, level, insert, guide))
    np.testing.assert_array_equal(
        out[np.asarray(level) == 0], np.asarray(image)[np.asarray(level) == 0]
    )
    # A trained branch must really see the original rough guide.
    a = model(image, level, insert, guide)
    b = model(image, level, insert, -guide)
    assert np.max(np.abs(np.asarray(a - b))) > 1e-7


def test_detail_requires_guided_unet():
    for kwargs in ({}, {"architecture": "nested", "source_conditioning": True}):
        with pytest.raises(ValueError, match="detail_channels requires"):
            ModelConfig(detail_channels=8, **kwargs)
    with pytest.raises(ValueError):
        ModelConfig(detail_channels=-1, source_conditioning=True)


def test_nonzero_detail_exports_dynamic_bgra_and_protects_pixels(tmp_path):
    c = configuration(8)
    model = create_model(c.model, rngs=nnx.Rngs(7))
    model.detail_head.layers[-1].kernel[...] = 0.01
    model.detail_head.layers[-1].bias[...] = -0.02
    runtime = LineartModel(export_model(model, c, tmp_path / "model.onnx"), provider="cpu")
    h, w = 145, 129
    scribble = np.zeros((h, w, 4), np.uint8)
    scribble[:, 35:38, 3] = 80
    lineart = np.zeros_like(scribble)
    lineart[30:33, :, 3] = 255
    plan = pyramid_plan(h, w, c.model.minimum_size)
    source, level = prepare_input(
        scribble.tobytes(), lineart.tobytes(), plan, 0.5, c.data.image_scale
    )
    inputs = runtime_inputs(source, level, 42, len(plan.sizes))
    for steps in (1, 2, 5):
        actual = runtime.infer(
            scribble=scribble.tobytes(),
            lineart=lineart.tobytes(),
            width=w,
            height=h,
            strength=0.5,
            seed=42,
            denoise_steps=steps,
        )
        pred = np.asarray(inference_with_source(model, *inputs, source, steps=steps))
        np.testing.assert_array_equal(pred[level == 0], source[level == 0])
        expected = to_bgra(pred, plan)
        error = np.abs(
            np.frombuffer(actual, np.uint8).astype(int)
            - np.frombuffer(expected, np.uint8).astype(int)
        )
        assert error.max() <= 1
