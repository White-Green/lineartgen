import numpy as np
from merge_family_data import merge_geometry as legacy_geometry
from support_merge_data import merge_allowed, merge_geometry


def record(separation):
    x = np.linspace(50, 150, 129)
    paths = [
        dict(points=np.stack((x, np.full_like(x, y)), -1).tolist(), widths=[2.0] * len(x))
        for y in (80 - separation / 2, 80 + separation / 2)
    ]
    return dict(kind="parallel", metadata=dict(path_geometry=paths))


def test_below_cutoff_geometry_is_unchanged():
    for sep in (2.0, 7.9, 8.0):
        a, b = legacy_geometry(record(sep)), merge_geometry(record(sep))
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])


def test_support_accepts_a_filled_band_but_rejects_disjoint_strokes():
    geo = merge_geometry(record(10.0))
    assert geo["merge_eligible"] and not legacy_geometry(record(10.0))["merge_eligible"]
    y, _ = np.indices((256, 256))
    band = np.exp(-(((y - 112) / 8) ** 8))
    separate = np.exp(-0.5 * ((y - 107) / 0.8) ** 2) + np.exp(-0.5 * ((y - 117) / 0.8) ** 2)
    for opacity in (0.2, 0.4, 0.8):
        assert merge_allowed((1 - 2 * opacity * band)[..., None], geo)
        assert not merge_allowed((1 - 2 * opacity * separate)[..., None], geo)


def test_one_crossing_bridge_does_not_join_entire_pair():
    geo = merge_geometry(record(10.0))
    y, x = np.indices((256, 256))
    lines = np.exp(-0.5 * ((y - 107) / 0.8) ** 2) + np.exp(-0.5 * ((y - 117) / 0.8) ** 2)
    bridge = np.exp(-0.5 * ((x - 132) / 2) ** 2) * (abs(y - 112) <= 7)
    source = (1 - 0.8 * np.maximum(lines, bridge))[..., None]
    assert not merge_allowed(source, geo)
