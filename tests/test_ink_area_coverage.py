import numpy as np
import pytest
from scipy import ndimage

from lineartgen.config import GeometryDataConfig
from lineartgen.data.coverage import COVERAGE_KINDS, draw_coverage
from lineartgen.data.geometry import mix_geometry


@pytest.mark.parametrize("kind", COVERAGE_KINDS)
def test_coverage_targets_are_reproducible_and_have_expected_interiors(kind):
    config = GeometryDataConfig()
    a, metadata = draw_coverage(np.random.default_rng(61), 256, config, kind=kind)
    b, same = draw_coverage(np.random.default_rng(61), 256, config, kind=kind)
    np.testing.assert_array_equal(a, b)
    assert metadata == same and metadata["coverage"]
    assert a.shape == (256, 256, 1) and a.dtype == np.float32
    assert np.isfinite(a).all() and 0 <= a.min() <= a.max() <= 1
    if kind in ("paper", "black"):
        np.testing.assert_array_equal(a, float(kind == "paper"))
    else:
        assert a.min() == 0 and a.max() == 1
        assert np.any((a > 0) & (a < 1))
        assert ndimage.distance_transform_edt(a[..., 0] < 0.02).max() > 8


def test_coverage_preserves_original_geometry_and_is_indexed_without_mutation():
    clean = np.random.default_rng(29).uniform(-1, 1, (16, 64, 64, 1)).astype(np.float32)
    before = clean.copy()
    old = GeometryDataConfig(probability=0.2)
    enabled = old.model_copy(update={"coverage_probability": 1.0})
    reference, old_records = mix_geometry(clean, old, seed=45, step=1, image_scale=1.1)
    result, records = mix_geometry(clean, enabled, seed=45, step=1, image_scale=1.1)
    repeated, same = mix_geometry(clean, enabled, seed=45, step=1, image_scale=1.1)
    np.testing.assert_array_equal(result, repeated)
    np.testing.assert_array_equal(clean, before)
    assert same == records and len(records) == len(clean)
    assert [r for r in records if not r.get("coverage")] == old_records
    geometry_items = [r["item"] for r in old_records]
    np.testing.assert_array_equal(result[geometry_items], reference[geometry_items])
    assert len({r["item"] for r in records}) == len(clean)
    for row in records:
        if row.get("coverage"):
            gray, metadata = draw_coverage(np.random.default_rng(row["key"]), 64, enabled)
            np.testing.assert_array_equal(result[row["item"]], np.clip((gray * 2 - 1) * 1.1, -1, 1))
            assert all(row[k] == v for k, v in metadata.items())
    only_coverage = GeometryDataConfig(probability=0, coverage_probability=1)
    _, rows = mix_geometry(clean, only_coverage, seed=45, step=1, image_scale=1.1)
    assert len(rows) == len(clean) and all(r["coverage"] for r in rows)


def test_coverage_width_validation():
    with pytest.raises(ValueError, match="Coverage width"):
        GeometryDataConfig(coverage_minimum_width=128, coverage_maximum_width=16)
