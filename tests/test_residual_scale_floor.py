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
from lineartgen.models.output import denoised_output
from lineartgen.models.serialization import load_inference, save_inference


def test_scale_floor_bounds_required_head_and_protects_zero_gradient():
    t = jnp.array([0, 0.001, 0.01, 0.05, 0.1, 0.5], dtype=jnp.float32)
    image = jnp.full_like(t, 0.2)
    head = jnp.full_like(t, -0.3)

    def call(h):
        return denoised_output(h, image, t, "residual", scale_floor=0.05)

    expected = image + jnp.where(t > 0, jnp.sqrt(jnp.maximum(t, 0.05)) * head, 0)
    np.testing.assert_array_equal(call(head), expected)
    np.testing.assert_allclose(
        jax.grad(lambda h: call(h).sum())(head),
        [0, *np.sqrt([0.05, 0.05, 0.05, 0.1, 0.5])],
        atol=1e-7,
    )
    assert call(head)[0] == image[0]


def test_default_is_bitwise_legacy_and_invalid_settings_fail():
    image = jnp.linspace(-1, 1, 32)
    t = jnp.linspace(0, 0.9, 32)
    head = jnp.linspace(-4, 4, 32)

    def old(h):
        return jnp.clip(image + jnp.sqrt(t) * h, -1, 1)

    def new(h):
        return denoised_output(h, image, t, "residual", scale_floor=0)

    np.testing.assert_array_equal(new(head), old(head))
    np.testing.assert_array_equal(
        jax.grad(lambda h: new(h).sum())(head), jax.grad(lambda h: old(h).sum())(head)
    )
    for kwargs in (
        {"output_mode": "direct"},
        {"architecture": "nested", "output_mode": "residual"},
    ):
        with pytest.raises(ValueError, match="requires a residual UNet"):
            ModelConfig(residual_scale_floor=0.05, **kwargs)


def test_high_strength_equivalence_and_serialization(tmp_path):
    c = ExperimentConfig.model_validate(
        dict(
            model=dict(
                channels=[2, 4],
                convolutions_per_block=2,
                output_mode="residual",
                source_conditioning=True,
            )
        )
    )
    base = create_model(c.model, rngs=nnx.Rngs(7))
    changed = nnx.clone(base)
    changed.residual_scale_floor = 0.05
    source = np.random.default_rng(3).uniform(-1, 1, (1, 32, 32, 1)).astype(np.float32)
    # Aligned protected region keeps every coarser map either zero or >=floor.
    strength = np.full_like(source, 0.1)
    strength[:, :8] = 0
    inputs = runtime_inputs(source, strength, 10, 3)
    np.testing.assert_array_equal(
        inference_with_source(base, *inputs, source, steps=1),
        inference_with_source(changed, *inputs, source, steps=1),
    )
    for steps in (1, 2, 5):
        out = np.asarray(inference_with_source(changed, *inputs, source, steps=steps))
        np.testing.assert_array_equal(out[strength == 0], source[strength == 0])
    c = c.model_copy(update={"model": c.model.model_copy(update={"residual_scale_floor": 0.05})})
    save_inference(changed, c, tmp_path / "weights", provenance="fixture")
    restored, config = load_inference(tmp_path / "weights")
    assert restored.residual_scale_floor == config.model.residual_scale_floor == 0.05
    jax.tree.map(np.testing.assert_array_equal, nnx.state(changed), nnx.state(restored))


def test_floored_model_exports_dynamic_bgra_and_preserves_protection(tmp_path):
    c = ExperimentConfig.model_validate(
        dict(
            model=dict(
                channels=[2, 4],
                convolutions_per_block=2,
                output_mode="residual",
                source_conditioning=True,
                residual_scale_floor=0.05,
            )
        )
    )
    model = create_model(c.model, rngs=nnx.Rngs(7))
    model.guide_encoder.kernel[...] = 0.02
    model.guide_head.kernel[...] = 0.01
    runtime = LineartModel(export_model(model, c, tmp_path / "model.onnx"), provider="cpu")
    h, w = 145, 129
    scribble = np.zeros((h, w, 4), np.uint8)
    scribble[:, 35:38, 3] = 80
    lineart = np.zeros_like(scribble)
    lineart[30:33, :, 3] = 255
    plan = pyramid_plan(h, w, c.model.minimum_size)
    source, level = prepare_input(
        scribble.tobytes(), lineart.tobytes(), plan, 0.01, c.data.image_scale
    )
    inputs = runtime_inputs(source, level, 42, len(plan.sizes))
    for steps in (1, 2, 5):
        actual = runtime.infer(
            scribble=scribble.tobytes(),
            lineart=lineart.tobytes(),
            width=w,
            height=h,
            strength=0.01,
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
