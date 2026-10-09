import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from lineartgen_runtime import LineartModel
from lineartgen_runtime.images import prepare_input, pyramid_plan, to_bgra
from lineartgen_runtime.sampling import denoise

from lineartgen.config import ExperimentConfig, ModelConfig
from lineartgen.diffusion import pyramid, recursive_forward, sample_prepared, training_inputs
from lineartgen.evaluation_sampling import inference_with_source, runtime_inputs
from lineartgen.models import create_model
from lineartgen.models.serialization import load_inference, save_inference
from lineartgen.sketch import augment_sketch
from lineartgen.training.lineart_steps import objective
from lineartgen.training.lineart_study import assert_initial_parameters, create_arm


def config():
    return ExperimentConfig.model_validate(
        dict(
            model=dict(
                channels=[2, 4],
                convolutions_per_block=2,
                output_mode="residual",
                source_conditioning=True,
            ),
            sketch=dict(mode="blur", probability=1),
            matching=dict(method="independent"),
            diffusion=dict(pattern_weights=[0, 0, 1, 0], identity_probability=0),
        )
    )


def test_guide_is_rough_source_and_existing_noisy_inputs_are_unchanged():
    c = config()
    clean = jnp.ones((2, 32, 32, 1)).at[:, :, 14:17].set(-1)
    key = jax.random.PRNGKey(9)
    old = training_inputs(clean, key, c.diffusion, c.matching, levels=3, sketch=c.sketch)
    guided = training_inputs(
        clean, key, c.diffusion, c.matching, levels=3, sketch=c.sketch, with_source=True
    )
    jax.tree.map(np.testing.assert_array_equal, tuple(old), tuple(guided[:3]))
    rough = augment_sketch(clean, jax.random.fold_in(key, 0x534B4554), c.sketch)
    jax.tree.map(np.testing.assert_array_equal, guided.guides, pyramid(rough, 3))
    assert not np.array_equal(rough, clean)
    assert not np.array_equal(guided.guides[0], guided.targets[0])
    assert not np.array_equal(guided.guides[0], guided.noisy[0])


def test_zero_projections_copy_parameters_and_preserve_recursive_outputs(tmp_path):
    c = config()
    oldc = c.model_copy(update={"model": c.model.model_copy(update={"source_conditioning": False})})
    old = create_model(oldc.model, rngs=nnx.Rngs(42))
    guided = create_arm(c, old)["model"]
    assert_initial_parameters(old, guided)
    rng = np.random.default_rng(18)
    source = rng.uniform(-1, 1, (1, 32, 32, 1)).astype(np.float32)
    noise = np.full_like(source, 0.5)
    noise[:, :8] = 0
    inputs = runtime_inputs(source, noise, 45, 3)
    guides = pyramid(jnp.asarray(source), 3)
    for steps in [1, 2, 5]:
        np.testing.assert_array_equal(
            sample_prepared(old, *inputs, steps=steps),
            sample_prepared(guided, *inputs, steps=steps, guides=guides),
        )
    jax.tree.map(
        np.testing.assert_array_equal,
        recursive_forward(old, *inputs),
        recursive_forward(guided, *inputs, guides=guides),
    )
    save_inference(guided, c, tmp_path / "model", provenance="zero projection unit test")
    loaded, saved = load_inference(tmp_path / "model")
    assert saved.model.source_conditioning
    jax.tree.map(np.testing.assert_array_equal, nnx.state(guided), nnx.state(loaded))
    with pytest.raises(ValueError, match="rough guide"):
        recursive_forward(guided, *inputs)
    with pytest.raises(ValueError, match="requires architecture"):
        ModelConfig(architecture="nested", source_conditioning=True)


def test_new_projections_receive_gradient_and_a_learned_guide_affects_output():
    c = config()
    model = create_model(c.model, rngs=nnx.Rngs(4))
    state = nnx.state(model, nnx.Param)
    nnx.update(model, jax.tree.map(lambda x: jnp.full_like(x, 0.02), state))
    # Start only the new guide projections at zero; keep the old network active.
    model.guide_encoder.kernel[...] = 0
    model.guide_head.kernel[...] = 0
    clean = jnp.full((1, 16, 16, 1), 0.2)
    prepared = training_inputs(
        clean,
        jax.random.PRNGKey(3),
        c.diffusion,
        c.matching,
        levels=2,
        sketch=c.sketch,
        with_source=True,
    )
    gradient = nnx.grad(lambda m: objective(m, prepared, None, c)[0])(model)
    for name in ["guide_encoder", "guide_head"]:
        assert any(np.any(np.asarray(x) != 0) for x in jax.tree.leaves(gradient[name]))
    model.guide_head.kernel[...] = 0.1
    inputs = runtime_inputs(np.asarray(clean), np.full_like(clean, 0.5), 14, 2)
    a = sample_prepared(model, *inputs, steps=1, guides=pyramid(jnp.zeros_like(clean), 2))
    b = sample_prepared(model, *inputs, steps=1, guides=pyramid(jnp.ones_like(clean), 2))
    assert np.max(np.abs(a - b)) > 1e-4


def test_runtime_guide_remains_original_and_downsampled_at_every_iteration():
    source = np.linspace(-1, 1, 32 * 32, dtype=np.float32).reshape(1, 32, 32, 1)
    noise = np.full_like(source, 0.4)
    seen = []
    expected = pyramid(jnp.asarray(source), 3)

    def forward(image, level, insert, guide):
        seen.append(guide.copy())
        return np.clip(image * 0.5, -1, 1)

    denoise(forward, source, noise, seed=20, steps=5, levels=3, source_conditioning=True)
    assert len(seen) == 15
    for index, actual in enumerate(seen):
        np.testing.assert_array_equal(actual, expected[2 - index % 3])


@pytest.mark.parametrize(
    "residual_base,feature_input", [("noisy", "noisy"), ("source", "noisy"), ("source", "source")]
)
def test_guided_onnx_preserves_krita_api_and_matches_jax(tmp_path, residual_base, feature_input):
    pytest.importorskip("jax2onnx")
    from lineartgen.export import export_model

    c = config()
    c = c.model_copy(
        update={
            "model": c.model.model_copy(
                update={"residual_base": residual_base, "feature_input": feature_input}
            )
        }
    )
    model = create_model(c.model, rngs=nnx.Rngs(7))
    # Exercise nonzero learned branches so ignoring the guide cannot pass.
    model.guide_encoder.kernel[...] = 0.03
    model.guide_head.kernel[...] = 0.02
    path = export_model(model, c, tmp_path / "model.onnx")
    runtime = LineartModel(path, provider="cpu")
    assert runtime.source_conditioning
    height, width = 145, 129
    scribble = np.zeros((height, width, 4), np.uint8)
    scribble[:, 35:38, 3] = 120
    lineart = np.zeros_like(scribble)
    lineart[30:33, :, 3] = 255
    plan = pyramid_plan(height, width, c.model.minimum_size, max(128, c.model.minimum_size))
    source, noise = prepare_input(
        scribble.tobytes(), lineart.tobytes(), plan, 0.5, c.data.image_scale
    )
    prepared = runtime_inputs(source, noise, 42, len(plan.sizes))
    for steps in [1, 2, 5]:
        actual = runtime.infer(
            scribble=scribble.tobytes(),
            lineart=lineart.tobytes(),
            width=width,
            height=height,
            strength=0.5,
            seed=42,
            denoise_steps=steps,
        )
        prediction = inference_with_source(model, *prepared, source, steps=steps)
        expected = to_bgra(np.asarray(prediction), plan)
        # Quantization may amplify tiny backend differences by one alpha level.
        a = np.frombuffer(actual, np.uint8).astype(int)
        b = np.frombuffer(expected, np.uint8).astype(int)
        assert np.max(np.abs(a - b)) <= 1
        assert len(actual) == height * width * 4
