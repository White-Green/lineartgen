import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pydantic import ValidationError

from lineartgen import diffusion
from lineartgen.config import DiffusionConfig, load_config
from lineartgen.evaluation import edit_boundary_metrics
from lineartgen.masks import boundary_rectangle, rectangle_candidates, rectangle_crossing_scores
from lineartgen.training.comparison import comparison_configs


def test_rectangle_scores_match_pixel_pairs_and_preserve_size():
    source = np.random.default_rng(11).uniform(-1.1, 1.1, (5, 19, 27, 1)).astype(np.float32)
    mask = jnp.zeros_like(source, dtype=bool).at[:, 3:14, 5:18].set(True)
    bounds = np.asarray(rectangle_candidates(mask, jax.random.PRNGKey(22), 8))
    scores = np.asarray(rectangle_crossing_scores(source, bounds))
    ink = np.clip((1 - source[..., 0]) / 2, 0, 1)
    expected = np.zeros_like(scores)
    for b in range(5):
        for c, (y0, x0, y1, x1) in enumerate(bounds[b]):
            for y in (y0, y1):
                if 0 < y < 19:
                    expected[b, c] += np.minimum(ink[b, y - 1, x0:x1], ink[b, y, x0:x1]).sum()
            for x in (x0, x1):
                if 0 < x < 27:
                    expected[b, c] += np.minimum(ink[b, y0:y1, x - 1], ink[b, y0:y1, x]).sum()
    np.testing.assert_allclose(scores, expected, atol=5e-6)
    selected = np.asarray(boundary_rectangle(source, mask, jax.random.PRNGKey(22), 8))
    np.testing.assert_array_equal(selected.sum((1, 2, 3)), np.asarray(mask).sum((1, 2, 3)))
    for b in range(5):
        y0, x0, y1, x1 = bounds[b, scores[b].argmax()]
        want = np.zeros((19, 27, 1), bool)
        want[y0:y1, x0:x1] = True
        np.testing.assert_array_equal(selected[b], want)
    np.testing.assert_array_equal(boundary_rectangle(source, mask, jax.random.PRNGKey(22), 1), mask)
    np.testing.assert_array_equal(
        boundary_rectangle(jnp.ones_like(source), mask, jax.random.PRNGKey(22), 8), mask
    )
    with pytest.raises(ValidationError):
        DiffusionConfig(mask_sampling="boundary_rect")


def test_mask_variants_keep_strength_area_and_nonrectangle_patterns():
    c = DiffusionConfig(pattern_weights=(1, 1, 1, 0))
    x = jnp.ones((64, 64, 64, 1)).at[:, 31:33, 10:54].set(-1)
    key = jax.random.PRNGKey(3)
    a = np.asarray(diffusion.noise_map(x, key, c))
    b = np.asarray(
        diffusion.noise_map(x, key, c.model_copy(update={"mask_sampling": "boundary_rect"}))
    )
    np.testing.assert_array_equal(a.max((1, 2, 3)), b.max((1, 2, 3)))
    np.testing.assert_array_equal((a > 0).sum((1, 2, 3)), (b > 0).sum((1, 2, 3)))
    draw = np.asarray(jax.random.uniform(key, (64, 10, 1, 1, 1)))[:, :, 0, 0, 0]
    unchanged = (draw[:, 1] <= 2 / 3) | (draw[:, 8] < c.identity_probability)
    np.testing.assert_array_equal(a[unchanged], b[unchanged])
    assert np.any(a[~unchanged] != b[~unchanged])
    for weights in ((1, 0, 0, 0), (0, 1, 0, 0)):
        config = c.model_copy(update={"pattern_weights": weights})
        np.testing.assert_array_equal(
            diffusion.noise_map(x, key, config),
            diffusion.noise_map(
                x, key, config.model_copy(update={"mask_sampling": "boundary_rect"})
            ),
        )


def test_mask_variants_share_source_and_gaussian_draws(monkeypatch):
    variants = comparison_configs(load_config("configs/mask-sampling.toml"), study="mask_sampling")
    clean = jnp.ones((16, 64, 64, 1)).at[:, 25:28, :].set(-1)
    sources, normals = [], []
    augment, normal = diffusion.augment_sketch, jax.random.normal

    def source(*a, **kw):
        value = augment(*a, **kw)
        sources.append(np.asarray(value))
        return value

    def gaussian(*a, **kw):
        value = normal(*a, **kw)
        normals.append(np.asarray(value))
        return value

    monkeypatch.setattr(diffusion, "augment_sketch", source)
    monkeypatch.setattr(jax.random, "normal", gaussian)
    prepared, draws = [], []
    for c in variants.values():
        normals.clear()
        prepared.append(
            diffusion.training_inputs(
                clean, jax.random.PRNGKey(12), c.diffusion, c.matching, levels=6, sketch=c.sketch
            )
        )
        draws.append(normals.copy())
    np.testing.assert_array_equal(sources[0], sources[1])
    for a, b in zip(draws[0], draws[1], strict=True):
        np.testing.assert_array_equal(a, b)
    for prepared_input in prepared:
        for x, y, t in zip(
            prepared_input.noisy, prepared_input.targets, prepared_input.noise_levels, strict=True
        ):
            mask = np.asarray(t) == 0
            np.testing.assert_array_equal(np.asarray(x)[mask], np.asarray(y)[mask])


def test_boundary_metrics_detect_erasure_and_spill_but_ignore_protected_pixels():
    target = np.ones((32, 32, 1), np.float32)
    target[15:17] = -1
    strength = np.zeros_like(target)
    strength[:, 16:] = 0.1
    perfect = edit_boundary_metrics(target, target, strength)
    assert perfect["edit_boundary_ink_f1"] == 1
    assert perfect["edit_boundary_mae"] == 0
    assert perfect["edit_boundary_pixels"] == 32 * 3
    bad = target.copy()
    bad[15:17, 16:19] = 1
    erased = edit_boundary_metrics(bad, target, strength)
    assert erased["edit_boundary_ink_f1"] == 0 and erased["edit_boundary_ink_mae"] == 2
    bad = target.copy()
    bad[14:18, 16:19] = -1
    thick = edit_boundary_metrics(bad, target, strength)
    assert thick["edit_boundary_background_mae"] > 0 and thick["edit_boundary_ink_f1"] < 1
    bad = target.copy()
    bad[:, :16] = 0
    assert edit_boundary_metrics(bad, target, strength) == perfect
    assert (
        edit_boundary_metrics(target, target, np.ones_like(strength))["edit_boundary_mae"] is None
    )
    blank = edit_boundary_metrics(np.ones_like(target), np.ones_like(target), strength)
    assert blank["edit_boundary_ink_f1"] is None and blank["edit_boundary_background_mae"] == 0
