import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from lineartgen_runtime import LineartModel
from lineartgen_runtime.images import prepare_input, pyramid_plan, to_bgra
from lineartgen_runtime.sampling import denoise

from lineartgen.config import DiffusionConfig, ExperimentConfig, ModelConfig, NoiseMatchingConfig
from lineartgen.diffusion import bridge_state, pyramid, sample_prepared, training_inputs
from lineartgen.evaluation_sampling import inference_with_source, runtime_inputs
from lineartgen.export import export_model
from lineartgen.models import create_model
from lineartgen.models.serialization import load_inference, save_inference
from lineartgen.sketch import restoration_target
from lineartgen.training.lineart_study import assert_initial_parameters, create_arm
from lineartgen.training.steps import make_steps


def config(*, progress=True, path="bridge"):
    return ExperimentConfig.model_validate(
        dict(
            model=dict(
                channels=[2, 4],
                convolutions_per_block=2,
                output_mode="residual",
                source_conditioning=True,
                detail_channels=8,
                progress_conditioning=progress,
            ),
            diffusion=dict(
                training_path=path,
                progress_initial_probability=0,
                pattern_weights=[0, 0, 1, 0],
                identity_probability=0,
            ),
            matching=dict(method="independent"),
        )
    )


def test_explicit_first_state_matches_original_training_and_default_contract():
    c = config()
    clean = jnp.ones((3, 32, 32, 1)).at[:, :, 14:17].set(-1)
    rough = 1 - (1 - clean) * 0.3
    key = jax.random.PRNGKey(9)
    old = training_inputs(rough, key, DiffusionConfig(), c.matching, levels=3, with_source=True)
    old = old._replace(targets=pyramid(restoration_target(clean, rough, old.noise_levels[0]), 3))
    settings = DiffusionConfig(training_path="bridge", progress_initial_probability=1)
    actual = training_inputs(
        clean,
        key,
        settings,
        c.matching,
        levels=3,
        with_source=True,
        with_progress=True,
        paired_source=rough,
    )
    jax.tree.map(np.testing.assert_array_equal, tuple(actual[:4]), tuple(old))
    assert all(np.all(np.asarray(x) == 1) for x in actual.progress)


def test_paired_training_changes_only_path_and_protects_mask_boundaries():
    c = config()
    clean = jnp.ones((8, 32, 32, 1)).at[:, 5:28, 14:17].set(-1)
    rough = 1 - (1 - clean) * 0.3
    key = jax.random.PRNGKey(17)
    arguments = dict(levels=3, with_source=True, with_progress=True, paired_source=rough)
    bridge = training_inputs(clean, key, c.diffusion, c.matching, **arguments)
    legacy = training_inputs(
        clean,
        key,
        c.diffusion.model_copy(update={"training_path": "rough"}),
        c.matching,
        **arguments,
    )
    for field in ("targets", "noise_levels", "guides", "progress"):
        jax.tree.map(np.testing.assert_array_equal, getattr(bridge, field), getattr(legacy, field))
    assert np.max(np.abs(bridge.noisy[0] - legacy.noisy[0])) > 0.01
    for r, target, t, u, a, b in zip(
        bridge.guides,
        bridge.targets,
        bridge.noise_levels,
        bridge.progress,
        bridge.noisy,
        legacy.noisy,
        strict=True,
    ):
        s = t / u
        expected_delta = (jnp.sqrt(u) * jnp.sqrt(1 - s) - jnp.sqrt(1 - t)) * (r - target)
        np.testing.assert_allclose(a - b, expected_delta, atol=5e-7, rtol=5e-6)
        protected = np.asarray(t) == 0
        assert protected.any()
        np.testing.assert_array_equal(np.asarray(a)[protected], np.asarray(r)[protected])
        np.testing.assert_array_equal(np.asarray(target)[protected], np.asarray(r)[protected])
        assert np.asarray(u).min() >= 0.05 and np.asarray(u).max() <= 1


@pytest.mark.parametrize("steps", [1, 2, 5])
def test_oracle_bridge_matches_both_samplers_and_their_progress(steps):
    source = np.full((1, 32, 32, 1), 0.6, np.float32)
    clean = np.ones_like(source)
    clean[:, :, 13:17] = -1
    strength = np.full_like(source, 0.5)
    strength[:, :7] = 0
    strength[:, :, :11] = 0
    target = np.where(strength > 0, clean, source)
    images, masks = runtime_inputs(source, strength, 42, 3)
    guides, targets = pyramid(jnp.asarray(source), 3), pyramid(jnp.asarray(target), 3)
    noise = []
    rng = np.random.default_rng(42)
    for image in images:
        noise.append(rng.standard_normal(image.shape, dtype=np.float32))

    class Oracle:
        source_conditioning = progress_conditioning = True
        output_mode = "residual"

        def __init__(self):
            self.calls = 0

        def __call__(self, image, level, insert, guide, progress):
            scale = 2 - self.calls % 3
            u = np.float32((steps - self.calls // 3) / steps)
            np.testing.assert_array_equal(progress, np.full_like(image, u))
            np.testing.assert_array_equal(guide, guides[scale])
            expected = bridge_state(guides[scale], targets[scale], masks[scale], noise[scale], u)
            np.testing.assert_allclose(image, expected, atol=5e-7, rtol=5e-6)
            self.calls += 1
            return targets[scale]

    oracle = Oracle()
    actual = sample_prepared(oracle, images, masks, steps=steps, guides=guides)
    np.testing.assert_array_equal(actual, target)
    assert oracle.calls == steps * 3
    oracle = Oracle()
    actual = denoise(
        oracle,
        source,
        strength,
        seed=42,
        steps=steps,
        levels=3,
        source_conditioning=True,
        progress_conditioning=True,
        preserve_zero_noise=True,
    )
    np.testing.assert_array_equal(actual, target)
    assert oracle.calls == steps * 3


def test_zero_progress_projections_preserve_function_and_serialize(tmp_path):
    old_config = config(progress=False, path="rough")
    old = create_model(old_config.model, rngs=nnx.Rngs(42))
    c = config()
    model = create_arm(c, old)["model"]
    assert_initial_parameters(old, model)
    assert_initial_parameters(old, create_model(c.model, rngs=nnx.Rngs(42)))
    source = np.random.default_rng(5).uniform(-1, 1, (1, 32, 32, 1)).astype(np.float32)
    strength = np.full_like(source, 0.5)
    strength[:, :7] = 0
    inputs = runtime_inputs(source, strength, 42, 3)
    for steps in (1, 2, 5):
        np.testing.assert_array_equal(
            inference_with_source(old, *inputs, source, steps=steps),
            inference_with_source(model, *inputs, source, steps=steps),
        )
    save_inference(model, c, tmp_path / "model", provenance="progress fixture")
    restored, saved = load_inference(tmp_path / "model")
    assert saved == c
    jax.tree.map(np.testing.assert_array_equal, nnx.state(model), nnx.state(restored))
    with pytest.raises(ValueError, match="matching progress"):
        model(jnp.asarray(source), jnp.asarray(strength), jnp.zeros((1, 16, 16, 1)), source)
    assert (
        sum(
            x.size
            for name in ("progress_encoder", "progress_head", "progress_detail")
            for x in jax.tree.leaves(nnx.state(model, nnx.Param)[name])
        )
        == 2 + 3 + 8
    )


def test_progress_receives_actual_loss_gradients_but_has_no_effect_at_initial_time():
    c = config()
    model = create_model(c.model, rngs=nnx.Rngs(4))
    nnx.update(model, jax.tree.map(lambda x: jnp.full_like(x, 0.02), nnx.state(model, nnx.Param)))
    names = ("progress_encoder", "progress_head", "progress_detail")
    for name in names:
        getattr(model, name).kernel[...] = 0
    clean = jnp.full((2, 16, 16, 1), 0.2)
    source = jnp.full_like(clean, 0.6)
    prepared = training_inputs(
        clean,
        jax.random.PRNGKey(3),
        c.diffusion,
        c.matching,
        levels=2,
        paired_source=source,
        with_source=True,
        with_progress=True,
    )
    gradient, _, _ = make_steps(c)
    metrics, grads = gradient(model, prepared)
    assert np.isfinite(float(metrics["loss"]))
    for name in names:
        assert any(np.any(np.asarray(x) != 0) for x in jax.tree.leaves(grads[name]))
    level = jnp.full_like(source, 0.3).at[:, :4].set(0)
    insert = jnp.zeros((2, 8, 8, 1))
    before = model(source, level, insert, source, jnp.ones_like(source))
    for name in names:
        getattr(model, name).kernel[...] = 0.1
    np.testing.assert_array_equal(
        before, model(source, level, insert, source, jnp.ones_like(source))
    )
    after = model(source, level, insert, source, jnp.full_like(source, 0.5))
    assert np.max(np.abs(np.asarray(after - before))) > 1e-5
    np.testing.assert_array_equal(
        np.asarray(after)[np.asarray(level) == 0], np.asarray(source)[np.asarray(level) == 0]
    )


def test_invalid_progress_or_bridge_configuration_is_rejected():
    for kwargs in (
        {},
        {"source_conditioning": True},
        {"source_conditioning": True, "output_mode": "residual", "architecture": "nested"},
    ):
        with pytest.raises(ValueError, match="progress_conditioning requires"):
            ModelConfig(progress_conditioning=True, **kwargs)
    with pytest.raises(ValueError, match="bridge training requires"):
        ExperimentConfig.model_validate(dict(diffusion=dict(training_path="bridge")))
    c = config()
    invalid = c.model_dump()
    invalid["matching"]["method"] = "exact"
    with pytest.raises(ValueError, match="independent Gaussian"):
        ExperimentConfig.model_validate(invalid)
    with pytest.raises(ValueError, match="Bridge training requires progress"):
        training_inputs(
            jnp.ones((1, 16, 16, 1)),
            jax.random.PRNGKey(0),
            c.diffusion,
            NoiseMatchingConfig(method="independent"),
            levels=1,
        )


def test_nonzero_progress_exports_and_preserves_bgra_contract(tmp_path):
    c = config()
    model = create_model(c.model, rngs=nnx.Rngs(7))
    for name in ("progress_encoder", "progress_head", "progress_detail"):
        getattr(model, name).kernel[...] = 0.03
    model.detail_head.layers[-1].kernel[...] = 0.01
    model.detail_head.layers[-1].bias[...] = -0.02
    runtime = LineartModel(export_model(model, c, tmp_path / "model.onnx"), provider="cpu")
    assert runtime.progress_conditioning
    height, width = 145, 129
    scribble = np.zeros((height, width, 4), np.uint8)
    scribble[:, 35:38, 3] = 80
    lineart = np.zeros_like(scribble)
    lineart[30:33, :, 3] = 255
    plan = pyramid_plan(height, width, c.model.minimum_size)
    source, strength = prepare_input(
        scribble.tobytes(), lineart.tobytes(), plan, 0.5, c.data.image_scale
    )
    inputs = runtime_inputs(source, strength, 42, len(plan.sizes))
    for steps in (1, 2, 5):
        actual_float = denoise(
            runtime.forward,
            source,
            strength,
            seed=42,
            steps=steps,
            levels=len(plan.sizes),
            source_conditioning=True,
            progress_conditioning=True,
            preserve_zero_noise=True,
        )
        expected = np.asarray(inference_with_source(model, *inputs, source, steps=steps))
        np.testing.assert_allclose(actual_float, expected, atol=2e-5, rtol=2e-4)
        np.testing.assert_array_equal(actual_float[strength == 0], source[strength == 0])
        actual = runtime.infer(
            scribble=scribble.tobytes(),
            lineart=lineart.tobytes(),
            width=width,
            height=height,
            strength=0.5,
            seed=42,
            denoise_steps=steps,
        )
        assert actual == to_bgra(actual_float, plan)
        diff = np.abs(
            np.frombuffer(actual, np.uint8).astype(int)
            - np.frombuffer(to_bgra(expected, plan), np.uint8).astype(int)
        )
        assert diff.max() <= 1
