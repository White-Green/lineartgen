import itertools

import jax
import jax.numpy as jnp
import numpy as np
from coherent_path_objective import (
    PATHS,
    SAMPLES,
    envelope_cost,
    global_path_cost,
    path_dynamic_cost,
    profile_unary,
    structured_parts,
)

from lineartgen.diffusion import pyramid


def straight_fixture():
    h = 128
    candidates = np.ones((1, 9, h, h, 1), np.float32)
    # Half-pixel AA translations of an opaque 2px vertical stroke.
    xx = np.arange(h, dtype=np.float32) + 0.5
    for i, shift in enumerate(np.linspace(-2, 2, 9)):
        coverage = np.maximum(
            0, np.minimum(xx + 0.5, 65 + shift) - np.maximum(xx - 0.5, 63 + shift)
        )
        candidates[0, i, 12:116, :, 0] = 1 - 2 * coverage
    probes = np.zeros((1, PATHS, SAMPLES, 6), np.float32)
    probes[..., 4] = 10
    probes[0, 0, :, 0] = 64
    probes[0, 0, :, 1] = np.linspace(16, 111, SAMPLES)
    probes[0, 0, :, 2] = 1
    valid = np.zeros((1, PATHS, SAMPLES), np.float32)
    valid[0, 0] = 1
    ann = dict(
        dense_probes=jnp.asarray(probes.reshape(1, -1, 6)),
        dense_valid=jnp.asarray(valid.reshape(1, -1)),
        spacing=jnp.full((1, PATHS, SAMPLES - 1), 95 / (SAMPLES - 1)),
    )
    return jnp.asarray(candidates), ann


def test_dp_matches_exhaustive_paths_and_breaks_at_protected_gap():
    rng = np.random.default_rng(72)
    unary = rng.uniform(size=(1, 1, 4, 9)).astype(np.float32)
    spacing = np.array([[[2, 3, 4]]], np.float32)
    offsets = np.linspace(-2, 2, 9)
    for mask in ([1, 1, 1, 1], [1, 0, 1, 1]):
        valid = np.array([[mask]], np.float32)
        expected = min(
            (
                sum(unary[0, 0, i, k] * mask[i] for i, k in enumerate(path))
                + sum(
                    0.25
                    * (offsets[path[i]] - offsets[path[i - 1]]) ** 2
                    / spacing[0, 0, i - 1]
                    * mask[i]
                    * mask[i - 1]
                    for i in range(1, 4)
                )
            )
            / sum(mask)
            for path in itertools.product(range(9), repeat=4)
        )
        actual = path_dynamic_cost(jnp.asarray(unary), jnp.asarray(valid), jnp.asarray(spacing))
        np.testing.assert_allclose(actual, expected, atol=2e-7)


def test_tie_gradient_selects_one_path_instead_of_gray_average():
    unary = jnp.zeros((1, 1, 5, 9))
    valid = jnp.ones((1, 1, 5))
    spacing = jnp.ones((1, 1, 4))
    grad = jax.grad(lambda u: path_dynamic_cost(u, valid, spacing).sum())(unary)
    np.testing.assert_allclose(grad[..., 0], 0.2)
    np.testing.assert_array_equal(grad[..., 1:], 0)


def test_local_choice_tolerates_smooth_drift_and_penalizes_switches():
    valid = jnp.ones((1, 1, 65))
    spacing = jnp.full((1, 1, 64), 2.0)

    def costs(states):
        return jnp.ones((1, 1, 65, 9)).at[0, 0, jnp.arange(65), jnp.asarray(states)].set(0)

    smooth = costs([4] * 13 + [5] * 13 + [6] * 13 + [5] * 13 + [4] * 13)
    abrupt = costs([0 if i % 2 == 0 else 8 for i in range(65)])
    assert float(path_dynamic_cost(smooth, valid, spacing)[0]) < 0.005
    assert float(global_path_cost(smooth, valid)[0]) > 0.2
    assert float(path_dynamic_cost(abrupt, valid, spacing)[0]) > 0.2


def test_actual_raster_profile_rejects_gray_missing_cut_and_outside_line():
    c, ann = straight_fixture()
    t = jnp.full_like(c[:, 0], 0.5)
    good = c[:, 4]
    gray = c.mean(1)
    missing = jnp.ones_like(good)
    cut = good.at[:, 52:76].set(1)
    outside = jnp.roll(good, 12, axis=2)
    for depth in (1, 3, 6):
        ts = pyramid(t, depth)
        results = {}
        for name, p in dict(
            good=good, gray=gray, missing=missing, cut=cut, outside=outside
        ).items():
            raw = pyramid(p, depth)
            parts = structured_parts(raw, c, ts, ann, "path")
            results[name] = float(parts["loss"])
        assert results["good"] < 1e-7
        assert all(
            results[k] > results["good"] + 1e-4 for k in ("gray", "missing", "cut", "outside")
        )
    # Wrong white saturation must receive a real gradient towards the permitted stroke.
    raw = jnp.full_like(good, 1.4)
    grad = jax.grad(lambda z: structured_parts((z,), c, (t,), ann, "path")["loss"])(raw)
    assert float(grad.sum()) > 0


def test_protected_values_no_gradient_and_blank_black_finite():
    c, ann = straight_fixture()
    p = c.mean(1)
    t = jnp.full_like(p, 0.5).at[:, :, :64].set(0)
    grad = jax.grad(lambda z: structured_parts((z,), c, (t,), ann, "path")["loss"])(p)
    np.testing.assert_array_equal(grad[:, :, :64], 0)
    no_paths = ann | dict(dense_valid=jnp.zeros_like(ann["dense_valid"]))
    for value in (-1.0, 1.0):
        constant = jnp.full_like(c, value)
        parts = structured_parts((constant[:, 0],), constant, (t,), no_paths, "path")
        assert float(parts["loss"]) == 0
        assert all(np.isfinite(x) for x in parts.values())
    # A shifted permitted clean is accepted by the envelope; outside support is penalized.
    assert float(envelope_cost((c[:, 0],), c, (jnp.ones_like(t),))[0]) == 0
    assert float(envelope_cost((jnp.roll(c[:, 0], 20, 2),), c, (jnp.ones_like(t),))[0]) > 0.1


def test_scale_choices_are_aggregated_before_path_selection():
    c, ann = straight_fixture()
    good = c[:, 4]
    t = jnp.full_like(good, 0.5)
    unary, visible = profile_unary(pyramid(good, 6), c, pyramid(t, 6), ann)
    assert unary.shape == (1, PATHS, SAMPLES, 9)
    assert visible[0, 0].all()
    assert float(global_path_cost(unary, visible)[0]) < 1e-7


def test_real_generator_update_checkpoint_restore_and_export(tmp_path):
    from coherent_path_objective import structured_gradient
    from cue_capacity import forward, make_arm, prepare_inputs
    from flax import nnx
    from scale_style_balance_study import update
    from test_full_source_normalization import config

    from lineartgen.config import ModelConfig
    from lineartgen.models import create_model
    from lineartgen.models.serialization import load_inference, save_inference
    from lineartgen.training import checkpoints, lineart_state

    cfg, _ = config()
    cfg = cfg.model_copy(
        update=dict(
            model=ModelConfig.model_validate(
                cfg.model.model_dump()
                | dict(local_dilations=(1, 4, 16), pyramid_conditioning="fourier")
            )
        )
    )
    candidates, ann = straight_fixture()
    source = candidates.mean(1)
    t = jnp.full_like(source, 0.5).at[:, :, :32].set(0)
    prepared = prepare_inputs(candidates[:, 4], source, t, jax.random.PRNGKey(9), depth=3)
    initial = create_model(cfg.model, rngs=nnx.Rngs(8))
    for method in ("global", "path"):
        arm = make_arm(cfg, initial)
        arm["gradient"] = structured_gradient(method)
        before = np.asarray(arm["model"].head.kernel[...]).copy()
        result = update(arm, prepared, candidates, ann)
        assert result["gradient_norm"] > 0
        assert not np.array_equal(before, arm["model"].head.kernel[...])
        with checkpoints.manager(tmp_path / method / "state") as saved:
            lineart_state.save(saved, {method: arm}, 1)
            restored = make_arm(cfg, initial)
            restored["gradient"] = structured_gradient(method)
            assert lineart_state.restore(saved, {method: restored}, 1) == 1
        jax.tree.map(
            np.testing.assert_array_equal,
            lineart_state.payload({method: arm}, 1),
            lineart_state.payload({method: restored}, 1),
        )
        assert update(arm, prepared, candidates, ann) == update(restored, prepared, candidates, ann)
        jax.tree.map(
            np.testing.assert_array_equal,
            lineart_state.payload({method: arm}, 2),
            lineart_state.payload({method: restored}, 2),
        )
        save_inference(
            arm["model"], cfg, tmp_path / method / "raw", provenance="coherent path test"
        )
        loaded, actual_cfg = load_inference(tmp_path / method / "raw")
        assert actual_cfg == cfg
        jax.tree.map(
            np.testing.assert_array_equal,
            forward(loaded, prepared),
            forward(arm["model"], prepared),
        )
        pred = np.asarray(forward(loaded, prepared)[0][0])
        np.testing.assert_array_equal(
            pred[np.asarray(t) == 0], np.asarray(source)[np.asarray(t) == 0]
        )
