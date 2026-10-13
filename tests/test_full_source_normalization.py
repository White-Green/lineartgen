import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cue_capacity import forward, make_arm, prepare_inputs
from flax import nnx
from lineartgen_runtime.sampling import denoise, local_contrast_source
from style_family_objective import style_gradient
from style_family_study import update
from test_family_objective import candidates
from test_path_profile import annotation
from test_plausible_lineart import fixture
from test_source_family import source_config

from lineartgen.config import ModelConfig
from lineartgen.diffusion import pyramid, recursive_forward, source_guides
from lineartgen.export import export_model
from lineartgen.models import create_model
from lineartgen.models.local import local_contrast_cue
from lineartgen.models.serialization import load_inference, save_inference
from lineartgen.training import checkpoints, lineart_state


def config():
    c, p, _ = fixture()
    c = source_config(c)
    mc = ModelConfig.model_validate(
        c.model.model_dump() | dict(source_preprocess="local_peak_full")
    )
    return c.model_copy(update={"model": mc}), p


def test_full_normalization_precedes_resize_and_matches_numpy():
    c, _ = config()
    m = create_model(c.model, rngs=nnx.Rngs(8))
    ink = jnp.zeros((1, 128, 128, 1)).at[:, 16:112, 30:32].set(0.1)
    source = 1 - 2 * ink
    guides = pyramid(source, 6)
    actual = source_guides(m, guides)
    expected = pyramid(local_contrast_cue(source), 6)
    jax.tree.map(np.testing.assert_array_equal, actual, expected)
    assert not np.allclose(actual[-1], local_contrast_cue(guides[-1]))
    np.testing.assert_allclose(local_contrast_source(np.asarray(source)), expected[0], atol=2e-7)
    scaled = source_guides(m, pyramid(1 - ink, 6))
    jax.tree.map(lambda x, y: np.testing.assert_allclose(x, y, atol=2e-7), actual, scaled)
    np.testing.assert_array_equal(local_contrast_source(np.ones((1, 4, 4, 1), np.float32)), 1)


def test_initial_capacity_functions_and_nonzero_head_opacity_invariance():
    c, p = config()
    mc = ModelConfig.model_validate(c.model.model_dump() | dict(source_preprocess="raw"))
    raw, normalized = [create_model(x, rngs=nnx.Rngs(8)) for x in (mc, c.model)]
    jax.tree.map(
        np.testing.assert_array_equal, nnx.state(raw, nnx.Param), nnx.state(normalized, nnx.Param)
    )
    assert sum(x.size for x in jax.tree.leaves(nnx.state(normalized, nnx.Param))) == 15809
    assert source_guides(raw, p.guides) is p.guides
    assert not np.array_equal(forward(raw, p)[0][0], forward(normalized, p)[0][0])
    normalized.head.kernel[...] = jnp.full_like(normalized.head.kernel[...], 0.01)
    ink = jnp.zeros((1, 128, 128, 1)).at[:, 16:112, 30:32].set(0.1)
    predictions = []
    for factor in (1.0, 0.5):
        src = 1 - 2 * factor * ink
        prepared = prepare_inputs(src, src, jnp.full_like(src, 0.5), jax.random.PRNGKey(2), depth=3)
        pred = forward(normalized, prepared)[0]
        direct = recursive_forward(
            normalized,
            prepared.noisy,
            prepared.noise_levels,
            guides=prepared.guides,
            progress=prepared.progress,
        )
        jax.tree.map(lambda x, y: np.testing.assert_allclose(x, y, atol=2e-7), pred, direct)
        predictions.append(pred)
    jax.tree.map(lambda x, y: np.testing.assert_allclose(x, y, atol=1e-6), *predictions)


def test_real_update_restore_export_and_protected_sampler(tmp_path):
    c, p = config()
    initial = create_model(c.model, rngs=nnx.Rngs(8))
    arm = make_arm(c, initial)
    arm["gradient"] = style_gradient()
    _, _, v = candidates()
    ann = annotation() | dict(normalizers=v)
    assert update(arm, p, v, ann)["gradient_norm"] > 0
    assert np.any(np.asarray(arm["model"].head.kernel[...]) != 0)
    with checkpoints.manager(tmp_path / "state") as saved:
        lineart_state.save(saved, {"normalized": arm}, 1)
        restored = make_arm(c, initial)
        restored["gradient"] = style_gradient()
        assert lineart_state.restore(saved, {"normalized": restored}, 1) == 1
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload({"normalized": arm}, 1),
        lineart_state.payload({"normalized": restored}, 1),
    )
    assert update(arm, p, v, ann) == update(restored, p, v, ann)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload({"normalized": arm}, 2),
        lineart_state.payload({"normalized": restored}, 2),
    )
    save_inference(
        arm["model"], c, tmp_path / "inference", provenance="full source normalization test"
    )
    loaded, cfg = load_inference(tmp_path / "inference")
    assert cfg == c and loaded.source_preprocess == "local_peak_full"
    jax.tree.map(np.testing.assert_array_equal, forward(loaded, p), forward(arm["model"], p))
    t = p.noise_levels[0].at[:, :, :16].set(0)
    prepared = prepare_inputs(p.targets[0], p.guides[0], t, jax.random.PRNGKey(8), depth=3)
    expected = forward(loaded, prepared)[0][0]
    np.testing.assert_array_equal(expected[t == 0], p.guides[0][t == 0])

    def stage(image, strength, insert, guide, progress):
        return np.asarray(loaded(*map(jnp.asarray, (image, strength, insert, guide, progress))))

    actual = denoise(
        stage,
        np.asarray(p.guides[0]),
        np.asarray(t),
        seed=10,
        steps=4,
        levels=3,
        source_conditioning=True,
        progress_conditioning=True,
        preserve_zero_noise=True,
        sampling_mode="one_pass",
        source_preprocess="local_peak_full",
    )
    np.testing.assert_allclose(actual, expected, atol=3e-7)
    np.testing.assert_array_equal(actual[t == 0], np.asarray(p.guides[0])[t == 0])


def test_invalid_configuration_and_export_rejected(tmp_path):
    c, _ = config()
    with pytest.raises(ValueError, match="Full source normalization"):
        ModelConfig(source_preprocess="local_peak_full")
    with pytest.raises(ValueError, match="Full source normalization"):
        ModelConfig.model_validate(c.model.model_dump() | dict(source_cue="local_peak"))
    with pytest.raises(ValueError, match="Full source normalization"):
        export_model(None, c, tmp_path / "unused.onnx")
