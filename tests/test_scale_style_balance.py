import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scale_style_balance_study as study
from cue_capacity import forward
from flax import nnx
from rough_family_data import BASE
from style_family_objective import style_gradient, style_path_costs, style_pixel_costs
from test_family_objective import candidates
from test_full_source_normalization import config
from test_path_profile import annotation

from lineartgen.config import ModelConfig
from lineartgen.models import create_model


def test_fourier_reuses_capacity_with_same_initial_function_and_real_level_response():
    cfg, prepared = config()
    base = create_model(cfg.model, rngs=nnx.Rngs(1336))
    mc = ModelConfig.model_validate(cfg.model.model_dump() | dict(pyramid_conditioning="fourier"))
    model = create_model(mc, rngs=nnx.Rngs(1336))
    jax.tree.map(
        np.testing.assert_array_equal, nnx.state(base, nnx.Param), nnx.state(model, nnx.Param)
    )
    assert sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param))) == 15809
    jax.tree.map(np.testing.assert_array_equal, forward(base, prepared), forward(model, prepared))
    model.head.kernel[...] = jnp.full_like(model.head.kernel[...], 0.01)
    _, _, v = candidates()
    _, grads = style_gradient()(model, prepared, v, annotation() | dict(normalizers=v))
    assert float(jnp.linalg.norm(grads["stem_modulation"]["kernel"][...])) > 0
    model.stem_modulation.kernel[...] = jnp.full_like(model.stem_modulation.kernel[...], 0.05)
    source = prepared.guides[0]
    args = (
        source,
        prepared.noise_levels[0],
        jnp.zeros((1, 16, 16, 1)),
        source,
        prepared.progress[0],
    )
    outputs = [model(*args, pyramid_level=k) for k in range(6)]
    assert np.max(abs(np.asarray(outputs[0] - outputs[5]))) > 1e-5
    assert len({np.asarray(v).tobytes() for v in outputs}) == 6
    with pytest.raises(ValueError, match="one-pass local"):
        ModelConfig(pyramid_conditioning="fourier")


def strokes():
    white = jnp.ones((1, 32, 32, 1))
    thin = white.at[:, :, 15:17].set(-1)
    thick = white.at[:, :, 8:24].set(-1)
    return white, thin[:, None], thick[:, None]


def test_clean_normalizer_penalizes_missing_thin_stroke_and_keeps_protection():
    white, thin, thick = strokes()
    t = (jnp.full_like(white, 0.5),)
    for fn in (
        lambda p, n: style_pixel_costs(p, thin, t, n),
        lambda p, n: style_path_costs(p, thin, t, annotation(), n),
    ):
        a, b = float(fn((white,), thick)[0, 0]), float(fn((white,), thin)[0, 0])
        assert b > 7 * a > 0
        assert float(fn((thin[:, 0],), thin).max()) == 0
    protected = jnp.full_like(white, 0.5).at[:, :, :16].set(0)
    grad = jax.grad(lambda p: style_pixel_costs((p,), thin, (protected,), thin).min())(white)
    np.testing.assert_array_equal(grad[:, :, :16], 0)
    for value in (-1.0, 1.0):
        clean = jnp.full_like(white, value)[:, None]
        cost = style_pixel_costs((clean[:, 0],), clean, t, clean)
        assert np.isfinite(cost).all() and float(cost.max()) == 0


def test_training_arms_share_sources_and_candidates_only_normalizer_changes(monkeypatch):
    white, thin, thick = strokes()
    original = jnp.repeat(thick, 27, axis=1)
    styled = jnp.repeat(thin, 27, axis=1)
    data = dict(
        variants=np.asarray(original),
        allstroke=np.asarray(styled),
        source=np.asarray(white),
        **annotation(),
    )
    monkeypatch.setattr(
        study,
        "selection",
        lambda step: dict(
            indices=[0] * 4, variants=[0] * 4, modes=[0] * 4, depth=1, augmentation=[BASE] * 4
        ),
    )
    monkeypatch.setattr(study, "strength_map", lambda mode: np.full((32, 32, 1), 0.5, np.float32))
    batches = {m: study.training_batch(data, 0, m) for m in ("initial",) + study.METHODS}
    for batch in batches.values():
        jax.tree.map(np.testing.assert_array_equal, batch[:2], batches["initial"][:2])
        for k in ("probes", "valid"):
            np.testing.assert_array_equal(batch[2][k], batches["initial"][2][k])
    np.testing.assert_array_equal(
        batches["initial"][2]["normalizers"], batches["level"][2]["normalizers"]
    )
    for m in ("balanced", "both"):
        np.testing.assert_array_equal(batches[m][2]["normalizers"], batches[m][1])
    assert not np.array_equal(batches["level"][2]["normalizers"], batches["both"][2]["normalizers"])
