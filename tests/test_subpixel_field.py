import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cue_capacity import make_arm, prepare_inputs
from flax import nnx
from opaque_geometry import render
from scale_style_balance_study import update
from subpixel_field_data import candidate_fields, central_field, geometry_field
from subpixel_field_model import (
    coverage,
    create_field_model,
    field_forward,
    infer,
    load_field,
    save_field,
)
from subpixel_field_objective import field_gradient, field_parts
from test_full_source_normalization import config

from lineartgen.config import ModelConfig
from lineartgen.diffusion import pyramid
from lineartgen.models.serialization import load_inference
from lineartgen.training import checkpoints, lineart_state


def fixture(size=32):
    seg = jnp.array([[[4.0, size / 2], [size - 4.0, size / 2]]])
    widths = jnp.array([[2.0, 2.0]])
    rect = jnp.empty((0, 4))
    field = geometry_field(seg, widths, rect, size=size)[None]
    return (seg, widths, rect), field


def test_field_decode_matches_opaque_oracle_and_geometry_plateau_is_penalized():
    args, base = fixture()
    np.testing.assert_allclose(
        coverage(central_field(base)[..., None])[0], render(*args, size=32), atol=2e-7
    )
    targets = candidate_fields(base, jnp.array([[0.0, 1.0]]), 3)
    fields = tuple(jnp.clip(x[:, 4], -8, 8) for x in targets)
    strengths = pyramid(jnp.full((1, 32, 32, 1), 0.5), 3)
    assert float(field_parts(fields, targets, strengths)["loss"]) < 1e-7
    gray = tuple(jnp.zeros_like(x) for x in fields)
    assert float(field_parts(gray, targets, strengths)["geometry_gradient"]) > 0
    assert float(field_parts(gray, targets, strengths)["loss"]) > 0.1
    grad = jax.grad(lambda f: field_parts((f,), targets[:1], strengths[:1])["loss"])(gray[0])
    assert float(jnp.linalg.norm(grad)) > 0
    wrong_white = jnp.full_like(fields[0], 12.0)
    g = jax.grad(lambda f: field_parts((f,), targets[:1], strengths[:1])["loss"])(wrong_white)
    assert float(g.sum()) > 0


def test_physical_geometry_sampling_all_six_levels_and_candidate_translation():
    seg = jnp.array([[[32.0, -64.0], [32.0, 128.0]]])
    w = jnp.array([[2.0, 2.0]])
    r = jnp.empty((0, 4))
    base = geometry_field(seg, w, r, size=64)[None]
    targets = candidate_fields(base, jnp.array([[1.0, 0.0]]), 6)
    for level, t in enumerate(targets):
        scale = 2**level
        x = (jnp.arange(256 // scale) + 0.5) * scale / 4
        for k, offset in enumerate(np.linspace(-2, 2, 9)):
            expected = jnp.clip((abs(x - (32 + offset)) - 1) / scale, -8, 8)
            actual = jnp.clip(t[0, k, 0, :, 0], -8, 8)
            # A quarter-pixel cache linearly interpolates the |x-center| cusp.
            # Its maximum center error is half a cache cell, scaled physically.
            assert float(jnp.max(abs(actual - expected))) <= 0.125 / scale + 2e-6
            away = abs(x - (32 + offset)) >= 0.125
            np.testing.assert_allclose(actual[away], expected[away], atol=2e-6)


def test_protected_geometry_loss_has_no_gradient():
    _, base = fixture()
    targets = candidate_fields(base, jnp.array([[0.0, 1.0]]), 1)
    strength = jnp.full((1, 32, 32, 1), 0.5).at[:, :, :16].set(0)
    p = jnp.zeros_like(targets[0][:, 4])
    g = jax.grad(lambda f: field_parts((f,), targets, (strength,))["loss"])(p)
    np.testing.assert_array_equal(g[:, :, :64], 0)


def test_real_update_restore_export_and_recursion(tmp_path):
    cfg, _ = config()
    cfg = cfg.model_copy(
        update=dict(
            model=ModelConfig.model_validate(
                cfg.model.model_dump()
                | dict(local_dilations=(1, 4, 16), pyramid_conditioning="fourier")
            )
        )
    )
    model = create_field_model(cfg, seed=8)
    assert sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param))) == 16064
    _, base = fixture()
    direction = jnp.array([[0.0, 1.0]])
    clean = coverage(central_field(base)[..., None])
    source = 1 - (1 - clean) * 0.4
    t = jnp.full_like(source, 0.5).at[:, :, :8].set(0)
    prepared = prepare_inputs(clean, source, t, jax.random.PRNGKey(3), 3)
    arm = make_arm(cfg, model)
    arm["gradient"] = field_gradient()
    assert update(arm, prepared, base, direction)["gradient_norm"] > 0
    assert np.any(np.asarray(arm["model"].head.kernel[...]) != 0)
    with checkpoints.manager(tmp_path / "state") as saved:
        lineart_state.save(saved, {"field": arm}, 1)
        restored = make_arm(cfg, model)
        restored["gradient"] = field_gradient()
        assert lineart_state.restore(saved, {"field": restored}, 1) == 1
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload({"field": arm}, 1),
        lineart_state.payload({"field": restored}, 1),
    )
    assert update(arm, prepared, base, direction) == update(restored, prepared, base, direction)
    save_field(arm["model"], cfg, tmp_path / "export", provenance="field test")
    loaded, cc = load_field(tmp_path / "export")
    assert cc == cfg
    np.testing.assert_array_equal(infer(loaded, prepared), infer(arm["model"], prepared))
    with pytest.raises(ValueError, match="Unsupported inference weight format"):
        load_inference(tmp_path / "export")
    pred = np.asarray(infer(loaded, prepared))
    np.testing.assert_array_equal(pred[np.asarray(t) == 0], np.asarray(source)[np.asarray(t) == 0])
    _, base64 = fixture(64)
    clean64 = coverage(central_field(base64)[..., None])
    s64 = 1 - (1 - clean64) * 0.4
    for depth in range(1, 7):
        p = prepare_inputs(clean64, s64, jnp.full_like(s64, 0.5), jax.random.PRNGKey(4), depth)
        fields = field_forward(loaded, p)
        assert len(fields) == depth
        assert [x.shape[1:3] for x in fields] == [(256 // 2**i, 256 // 2**i) for i in range(depth)]
        assert all(np.isfinite(x).all() for x in fields)
