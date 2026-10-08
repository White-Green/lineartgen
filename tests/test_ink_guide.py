import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from lineartgen_runtime import LineartModel
from lineartgen_runtime.images import prepare_input, pyramid_plan, to_bgra

from lineartgen.config import ExperimentConfig, ModelConfig
from lineartgen.evaluation_sampling import inference_with_source, runtime_inputs
from lineartgen.export import export_model
from lineartgen.models import create_model
from lineartgen.models.serialization import load_inference, save_inference
from lineartgen.models.unet import encode_guide
from lineartgen.training.lineart_study import assert_initial_parameters, create_arm


def config(encoding):
    return ExperimentConfig.model_validate(
        dict(
            model=dict(
                channels=[2, 4],
                convolutions_per_block=2,
                output_mode="residual",
                source_conditioning=True,
                guide_encoding=encoding,
            )
        )
    )


def test_white_zero_and_faint_signal_transform():
    values = jnp.asarray([-1.0, 0.5, 0.98, 1.0], jnp.float32)
    np.testing.assert_array_equal(encode_guide(values, "raw"), values)
    np.testing.assert_allclose(encode_guide(values, "ink"), [1, 0.25, 0.01, 0], atol=1e-7)
    np.testing.assert_allclose(encode_guide(values, "sqrt_ink"), [1, 0.5, 0.1, 0], atol=1e-7)
    with pytest.raises(ValueError, match="requires source_conditioning"):
        ModelConfig(guide_encoding="ink")


@pytest.mark.parametrize("encoding", ["ink", "sqrt_ink"])
def test_zero_initialization_preserves_function_and_learned_white_projection_is_zero(
    tmp_path, encoding
):
    c = config(encoding)
    base = create_model(
        ModelConfig(channels=(2, 4), convolutions_per_block=2, output_mode="residual"),
        rngs=nnx.Rngs(9),
    )
    model = create_arm(c, base)["model"]
    assert_initial_parameters(base, model)
    rng = np.random.default_rng(13)
    source = rng.uniform(-1, 1, (1, 32, 32, 1)).astype(np.float32)
    level = np.full_like(source, 0.1)
    level[:, :8] = 0
    prepared = runtime_inputs(source, level, 14, 3)
    for steps in (1, 2, 5):
        np.testing.assert_array_equal(
            inference_with_source(base, *prepared, source, steps=steps),
            inference_with_source(model, *prepared, source, steps=steps),
        )
    # Even after learning, the auxiliary branch cannot create a DC paper response.
    model.guide_encoder.kernel[...] = 0.02
    model.guide_head.kernel[...] = 0.03
    white = encode_guide(jnp.ones_like(source), encoding)
    assert np.count_nonzero(model.guide_encoder(white)) == 0
    assert np.count_nonzero(model.guide_head(white)) == 0
    save_inference(model, c, tmp_path / "model", provenance="fixture")
    loaded, restored = load_inference(tmp_path / "model")
    assert restored.model.guide_encoding == loaded.guide_encoding == encoding
    jax.tree.map(np.testing.assert_array_equal, nnx.state(model), nnx.state(loaded))


@pytest.mark.parametrize("encoding", ["ink", "sqrt_ink"])
def test_encoded_guide_exports_and_matches_krita_contract(tmp_path, encoding):
    c = config(encoding)
    model = create_model(c.model, rngs=nnx.Rngs(7))
    model.guide_encoder.kernel[...] = 0.03
    model.guide_head.kernel[...] = 0.02
    path = export_model(model, c, tmp_path / "model.onnx")
    runtime = LineartModel(path, provider="cpu")
    height, width = 145, 129
    scribble = np.zeros((height, width, 4), np.uint8)
    scribble[:, 35:38, 3] = 40
    lineart = np.zeros_like(scribble)
    lineart[30:33, :, 3] = 255
    plan = pyramid_plan(height, width, c.model.minimum_size)
    source, level = prepare_input(
        scribble.tobytes(), lineart.tobytes(), plan, 0.1, c.data.image_scale
    )
    prepared = runtime_inputs(source, level, 42, len(plan.sizes))
    for steps in (1, 2, 5):
        result = runtime.infer(
            scribble=scribble.tobytes(),
            lineart=lineart.tobytes(),
            width=width,
            height=height,
            strength=0.1,
            seed=42,
            denoise_steps=steps,
        )
        prediction = np.asarray(inference_with_source(model, *prepared, source, steps=steps))
        expected = to_bgra(prediction, plan)
        diff = np.abs(
            np.frombuffer(result, np.uint8).astype(int)
            - np.frombuffer(expected, np.uint8).astype(int)
        )
        assert diff.max() <= 1
