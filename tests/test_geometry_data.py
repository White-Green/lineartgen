import jax
import numpy as np
import pytest

from lineartgen.config import GeometryDataConfig, load_config
from lineartgen.data.geometry import KINDS, draw_geometry, mix_geometry
from lineartgen.diffusion import training_inputs
from lineartgen.training.comparison import comparison_configs


@pytest.mark.parametrize("kind", KINDS)
def test_continuous_geometry_is_deterministic_and_nonblank(kind):
    settings = GeometryDataConfig()
    a, metadata = draw_geometry(np.random.default_rng(52), 128, settings, kind=kind)
    b, same = draw_geometry(np.random.default_rng(52), 128, settings, kind=kind)
    c, _ = draw_geometry(np.random.default_rng(53), 128, settings, kind=kind)
    np.testing.assert_array_equal(a, b)
    assert metadata == same and metadata["kinds"] == [kind]
    assert a.shape == (128, 128, 1) and a.dtype == np.float32
    assert 0 <= a.min() < 0.5 and a.max() == 1
    assert 0.001 < np.mean(a < 0.5) < 0.5
    assert np.any((a > 0) & (a < 1)) and not np.array_equal(a, c)


def test_target_replacement_is_replayable_without_mutating_or_consuming_crop_rng():
    clean = np.random.default_rng(29).uniform(-1, 1, (16, 64, 64, 1)).astype(np.float32)
    original = clean.copy()
    disabled, records = mix_geometry(clean, GeometryDataConfig(), seed=45, step=1, image_scale=1)
    assert disabled is clean and not records
    settings = GeometryDataConfig(probability=0.2)
    mixed, records = mix_geometry(clean, settings, seed=45, step=1, image_scale=1)
    assert 0 < len(records) < len(clean)
    repeated, repeated_records = mix_geometry(clean, settings, seed=45, step=1, image_scale=1)
    np.testing.assert_array_equal(mixed, repeated)
    assert records == repeated_records
    np.testing.assert_array_equal(clean, original)
    changed = [r["item"] for r in records]
    np.testing.assert_array_equal(
        mixed[[i for i in range(16) if i not in changed]],
        clean[[i for i in range(16) if i not in changed]],
    )
    for record in records:
        generated, info = draw_geometry(
            np.random.default_rng(np.random.SeedSequence(record["key"])), 64, settings
        )
        np.testing.assert_array_equal(mixed[record["item"]], generated * 2 - 1)
        assert record["kinds"] == info["kinds"] and record["paths"] == info["paths"]
    other, _ = mix_geometry(clean, settings, seed=45, step=2, image_scale=1)
    assert not np.array_equal(mixed, other)
    blank, records = mix_geometry(
        clean,
        GeometryDataConfig(probability=1, blank_probability=1),
        seed=45,
        step=1,
        image_scale=1.5,
    )
    assert np.all(blank == 1) and all(r["kinds"] == ["blank"] for r in records)


def test_geometry_comparison_changes_only_target_source_and_keeps_shared_masks_and_inputs():
    config = load_config("configs/geometry-data.toml")
    variants = comparison_configs(config, study="geometry_data")
    control, geometry = variants.values()
    assert control == geometry.model_copy(update={"geometry": control.geometry})
    clean = np.random.default_rng(90).uniform(-1, 1, (16, 32, 32, 1)).astype(np.float32)
    mixed, records = mix_geometry(clean, geometry.geometry, seed=45, step=1, image_scale=1)
    key = jax.random.PRNGKey(37)
    a = training_inputs(
        clean, key, control.diffusion, control.matching, levels=2, sketch=control.sketch
    )
    b = training_inputs(
        mixed, key, geometry.diffusion, geometry.matching, levels=2, sketch=geometry.sketch
    )
    jax.tree.map(np.testing.assert_array_equal, a.noise_levels, b.noise_levels)
    unchanged = [i for i in range(16) if i not in [r["item"] for r in records]]
    for left, right in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_array_equal(np.asarray(left)[unchanged], np.asarray(right)[unchanged])
    for source, target, mask in zip(b.noisy, b.targets, b.noise_levels, strict=True):
        protected = np.asarray(mask) == 0
        np.testing.assert_array_equal(np.asarray(source)[protected], np.asarray(target)[protected])


def test_parallel_curves_and_intended_gap_remain_two_raster_components():
    from scipy.ndimage import label

    settings = GeometryDataConfig(minimum_width=2, maximum_width=12)
    for kind, index in (("parallel", 6), ("gap", 7)):
        gray, info = draw_geometry(
            np.random.default_rng(np.random.SeedSequence([45, index, 0x50524556])),
            512,
            settings,
            kind=kind,
        )
        _, count = label(gray[..., 0] < 0.5, structure=np.ones((3, 3)))
        assert count == 2 and info["paths"] == 2
