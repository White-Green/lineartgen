import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cue_capacity import forward, make_arm, prepare_inputs
from flax import nnx
from lineartgen_runtime.sampling import denoise
from local_contrast_family_study import merge_duplicate_features
from local_recursive_fit import fresh_model
from style_family_objective import style_gradient
from style_family_study import update
from test_family_objective import candidates
from test_path_profile import annotation
from test_plausible_lineart import fixture
from test_source_family import source_config

from lineartgen.config import ModelConfig
from lineartgen.export import export_model
from lineartgen.models import create_model
from lineartgen.models.local import local_contrast_cue
from lineartgen.models.serialization import load_inference, save_inference


def configs():
    c, p, _ = fixture()
    c = source_config(c)
    mc = ModelConfig.model_validate(c.model.model_dump() | dict(source_cue="local_peak"))
    return c, c.model_copy(update={"model": mc}), p


def models():
    c, local, p = configs()
    parent = fresh_model(c)
    parent.head.kernel[...] = jnp.full_like(parent.head.kernel[...], 0.01)
    raw = nnx.clone(parent)
    merge_duplicate_features(raw)
    normalized = create_model(local.model, rngs=nnx.Rngs(0))
    nnx.update(normalized, nnx.state(raw, nnx.Param))
    return parent, raw, normalized, c, local, p


def test_cue_is_local_bounded_and_opacity_invariant_above_floor():
    ink = jnp.zeros((1, 128, 128, 1)).at[:, 32:96, 16:20].set(0.1)
    ink = ink.at[:, 32:96, 96:100].set(0.8)
    a = local_contrast_cue(1 - 2 * ink)
    b = local_contrast_cue(1 - 2 * ink * 0.5)
    np.testing.assert_allclose(a, b, atol=2e-7)
    np.testing.assert_allclose(a[0, 64, [18, 98], 0], [0.2, 0.2], atol=1e-7)
    np.testing.assert_array_equal(a[ink == 0], 1)
    assert np.isfinite(np.asarray(a)).all() and np.min(a) >= 0.1999999
    blank = jnp.ones((1, 8, 8, 1))
    np.testing.assert_array_equal(local_contrast_cue(blank), blank)
    tiny = local_contrast_cue(blank.at[:, 3, 3].set(0.998))
    np.testing.assert_allclose(tiny[0, 3, 3, 0], 0.96, atol=2e-6)


def test_initial_transfer_capacity_and_serialization(tmp_path):
    parent, raw, local, _, cfg, p = models()
    original, a, b = [forward(m, p) for m in (parent, raw, local)]
    jax.tree.map(np.testing.assert_array_equal, a, b)
    jax.tree.map(lambda x, y: np.testing.assert_allclose(x, y, atol=2e-5, rtol=1e-5), original, a)
    for model in (raw, local):
        assert sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param))) == 15809
    save_inference(local, cfg, tmp_path / "local", provenance="contrast cue test")
    loaded, restored = load_inference(tmp_path / "local")
    assert cfg == restored and loaded.source_cue == "local_peak"
    jax.tree.map(np.testing.assert_array_equal, b, forward(loaded, p))


def test_cue_has_distinct_trainable_gradient_and_keeps_protection_and_sampler():
    _, raw, local, _, cfg, p = models()
    _, _, v = candidates()
    ann = annotation() | dict(normalizers=v)
    gradient = style_gradient()
    raw_parts, raw_grads = gradient(raw, p, v, ann)
    local_parts, local_grads = gradient(local, p, v, ann)
    jax.tree.map(np.testing.assert_array_equal, raw_parts, local_parts)
    assert not np.array_equal(
        np.asarray(raw_grads["stem"]["kernel"][:, :, 0, :]),
        np.asarray(local_grads["stem"]["kernel"][:, :, 0, :]),
    )
    arm = make_arm(cfg, local)
    local = arm["model"]
    arm["gradient"] = gradient
    assert update(arm, p, v, ann)["gradient_norm"] > 0
    assert np.any(np.asarray(local.stem.kernel[:, :, 0, :]) != 0)
    t = p.noise_levels[0].at[:, :, :16].set(0)
    prepared = prepare_inputs(p.targets[0], p.guides[0], t, jax.random.PRNGKey(9), depth=2)
    result = forward(local, prepared)[0][0]
    np.testing.assert_array_equal(result[t == 0], p.guides[0][t == 0])

    def stage(image, strength, insert, guide, progress):
        return np.asarray(local(*map(jnp.asarray, (image, strength, insert, guide, progress))))

    output = denoise(
        stage,
        np.asarray(p.guides[0]),
        np.asarray(t),
        seed=7,
        steps=3,
        levels=2,
        source_conditioning=True,
        progress_conditioning=True,
        preserve_zero_noise=True,
        sampling_mode="one_pass",
    )
    np.testing.assert_allclose(output, np.asarray(result), atol=2e-7)


def test_invalid_config_and_unvalidated_export_rejected(tmp_path):
    with pytest.raises(ValueError, match="Local contrast"):
        ModelConfig(source_cue="local_peak")
    _, cfg, _ = configs()
    with pytest.raises(ValueError, match="Local contrast"):
        export_model(None, cfg, tmp_path / "unused.onnx")
