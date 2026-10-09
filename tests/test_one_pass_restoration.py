import json

import jax
import numpy as np
import pytest
from flax import nnx
from lineartgen_runtime import LineartModel
from lineartgen_runtime.images import prepare_input, pyramid_plan, to_bgra
from lineartgen_runtime.sampling import denoise

from lineartgen.config import ExperimentConfig, ModelConfig
from lineartgen.diffusion import sample_prepared
from lineartgen.evaluation_sampling import inference_with_source, runtime_inputs
from lineartgen.export import export_model
from lineartgen.models import create_model


def config(mode="one_pass"):
    return ExperimentConfig(
        model=ModelConfig(
            channels=(2, 4),
            convolutions_per_block=2,
            source_conditioning=True,
            feature_input="source",
            residual_base="source",
            output_mode="residual",
            sampling_mode=mode,
        )
    )


def test_one_pass_preserves_requested_strength_and_protected_pixels():
    c = config()
    model = create_model(c.model, rngs=nnx.Rngs(7))
    source = np.linspace(-0.7, 0.7, 32 * 32, dtype=np.float32).reshape(1, 32, 32, 1)
    mask = np.full_like(source, 0.5)
    mask[:, :7] = 0
    arguments = runtime_inputs(source, mask, 31, 3)
    one = inference_with_source(model, *arguments, source, steps=1)
    for steps in (2, 5, 20):
        actual = inference_with_source(model, *arguments, source, steps=steps)
        np.testing.assert_array_equal(one, actual)
        np.testing.assert_array_equal(actual[:, :7], source[:, :7])
    weaker = inference_with_source(model, *runtime_inputs(source, mask / 5, 31, 3), source, steps=1)
    assert np.max(np.abs(one - weaker)) > 1e-3
    legacy = nnx.clone(model)
    legacy.sampling_mode = "diffusion"
    np.testing.assert_array_equal(one, inference_with_source(legacy, *arguments, source, steps=1))
    assert ModelConfig().sampling_mode == "diffusion"
    with pytest.raises(ValueError, match="requires source feature"):
        ModelConfig(sampling_mode="one_pass")
    with pytest.raises(ValueError, match="positive"):
        sample_prepared(model, *arguments, steps=0)


def test_numpy_one_pass_calls_each_resolution_once_at_original_strength():
    source = np.full((1, 32, 48, 1), 0.3, np.float32)
    mask = np.full_like(source, 0.5)
    mask[:, :7] = 0
    calls = []

    def forward(image, noise, insert, guide):
        calls.append((image.shape, float(noise.max())))
        return np.where(noise == 0, image, guide - 0.2 * np.sqrt(noise))

    actual = denoise(
        forward,
        source,
        mask,
        seed=31,
        steps=20,
        levels=3,
        source_conditioning=True,
        preserve_zero_noise=True,
        sampling_mode="one_pass",
    )
    assert calls == [((1, 8, 12, 1), 0.5), ((1, 16, 24, 1), 0.5), ((1, 32, 48, 1), 0.5)]
    np.testing.assert_array_equal(actual[:, :7], source[:, :7])
    assert np.max(np.abs(actual[:, 7:] - (source[:, 7:] - 0.2 * np.sqrt(0.5)))) < 1e-7


def test_one_pass_onnx_actual_bgra_and_metadata_contract(tmp_path):
    import onnx

    c = config()
    model = create_model(c.model, rngs=nnx.Rngs(7))
    path = export_model(model, c, tmp_path / "model.onnx")
    runtime = LineartModel(path, provider="cpu")
    assert runtime.sampling_mode == "one_pass"
    assert json.loads(path.with_suffix(".json").read_text())["format_version"] == 2
    width, height = 145, 129
    scribble = np.zeros((height, width, 4), np.uint8)
    scribble[:, 60:63, 3] = 100
    lineart = np.zeros_like(scribble)
    lineart[:31, :55] = [0, 0, 0, 255]
    kwargs = dict(
        scribble=scribble.tobytes(),
        lineart=lineart.tobytes(),
        width=width,
        height=height,
        strength=0.5,
        seed=31,
    )
    output = runtime.infer(**kwargs, denoise_steps=1)
    for steps in (2, 5, 20):
        assert runtime.infer(**kwargs, denoise_steps=steps) == output
    assert runtime.infer(**(kwargs | dict(seed=57)), denoise_steps=5) == output
    plan = pyramid_plan(height, width, c.model.minimum_size, 128)
    source, mask = prepare_input(
        kwargs["scribble"], kwargs["lineart"], plan, 0.5, c.data.image_scale
    )
    prepared = runtime_inputs(source, mask, 31, len(plan.sizes))
    prediction = np.asarray(
        inference_with_source(model, *jax.device_put(prepared), jax.device_put(source), steps=5)
    )
    np.testing.assert_array_equal(prediction[mask == 0], source[mask == 0])
    expected = np.frombuffer(to_bgra(prediction, plan), np.uint8).astype(int)
    assert np.max(np.abs(np.frombuffer(output, np.uint8).astype(int) - expected)) <= 1
    # An old-format label must not permit silent fallback to the diffusion loop.
    graph = onnx.load(path)
    metadata = json.loads(graph.metadata_props[0].value)
    metadata["format_version"] = 1
    onnx.helper.set_model_props(graph, {"lineartgen": json.dumps(metadata)})
    bad = tmp_path / "mislabeled.onnx"
    onnx.save(graph, bad)
    with pytest.raises(ValueError, match="format version 2"):
        LineartModel(bad, provider="cpu")
