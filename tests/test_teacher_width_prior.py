import numpy as np
import pytest

from lineartgen.config import GeometryDataConfig
from lineartgen.data.geometry import KINDS, draw_geometry, mix_geometry


@pytest.mark.parametrize("kind", KINDS)
def test_paired_width_change_preserves_draw_count_and_thin_kind(kind):
    old = GeometryDataConfig(
        probability=0.2, minimum_width=0.6, maximum_width=12, coverage_probability=0.125
    )
    new = old.model_copy(update={"maximum_width": 6.0})
    a, ma = draw_geometry(
        np.random.default_rng(2026101045), 128, old, kind=kind, include_paths=True
    )
    b, mb = draw_geometry(
        np.random.default_rng(2026101045), 128, new, kind=kind, include_paths=True
    )
    assert ma["kinds"] == mb["kinds"] and ma["paths"] == mb["paths"]
    for pa, pb in zip(ma["path_geometry"], mb["path_geometry"], strict=True):
        wa, wb = np.asarray(pa["widths"]), np.asarray(pb["widths"])
        assert np.all(wb <= wa) and np.max(wb) <= 6
        if kind not in ["parallel", "gap"]:
            np.testing.assert_array_equal(pa["points"], pb["points"])
        if kind == "thin":
            np.testing.assert_array_equal(wa, wb)
    if kind == "thin":
        np.testing.assert_array_equal(a, b)


def test_width_change_only_changes_geometry_items_not_native_or_coverage():
    old = GeometryDataConfig(
        probability=0.5, minimum_width=0.6, maximum_width=12, coverage_probability=0.5
    )
    new = old.model_copy(update={"maximum_width": 6.0})
    clean = np.random.default_rng(1).uniform(-1, 1, (4, 64, 64, 1)).astype(np.float32)
    counts = {"geometry": 0, "coverage": 0, "native": 0, "changed": 0}
    for step in range(12):
        a, ma = mix_geometry(
            clean, old, seed=2026100961, step=step, image_scale=1.1, include_paths=True
        )
        b, mb = mix_geometry(
            clean, new, seed=2026100961, step=step, image_scale=1.1, include_paths=True
        )
        assert [x["item"] for x in ma] == [x["item"] for x in mb]
        geom = {x["item"] for x in ma if not x.get("coverage")}
        coverage = {x["item"] for x in ma if x.get("coverage")}
        for i in range(4):
            key = "geometry" if i in geom else "coverage" if i in coverage else "native"
            counts[key] += 1
            if i not in geom:
                np.testing.assert_array_equal(a[i], b[i])
            else:
                counts["changed"] += int(np.any(a[i] != b[i]))
        for x, y in zip(ma, mb, strict=True):
            assert x["kinds"] == y["kinds"]
            if x.get("coverage"):
                assert x == y
            else:
                assert x["paths"] == y["paths"]
    assert all(counts.values())
