from collections import Counter

import numpy as np
from attraction_width_data import geometry, record


def test_width_recipes_replay_and_stay_inside_canvas():
    rows = [record(i) for i in range(1024)]
    assert Counter(r["kind"] for r in rows) == {
        "bar4-16": 512,
        "bar16-48": 256,
        "bar48-96": 128,
        "fill": 128,
    }
    for r in rows:
        assert r == record(r["index"])
        c, s = abs(np.cos(r["angle"])), abs(np.sin(r["angle"]))
        rx, ry = (c * r["width"] + s * r["length"]) / 2, (s * r["width"] + c * r["length"]) / 2
        x, y = r["center"]
        assert x - rx >= 8 - 1e-5 and x + rx <= 248 + 1e-5
        assert y - ry >= 8 - 1e-5 and y + ry <= 248 + 1e-5
        if not r["filled"]:
            assert r["length"] >= 1.05 * r["width"]


def test_graph_centerline_follows_source_axis_and_fill_uses_integer_bounds():
    for i in (0, 2, 513, 768, 769, 770, 896, 1023):
        r = record(i)
        segments, widths, rectangles = geometry(r)
        if r["filled"]:
            assert len(segments) == 0
            np.testing.assert_array_equal(rectangles, np.rint(rectangles))
            np.testing.assert_allclose(
                rectangles[0, 2:] - rectangles[0, :2], [r["width"], r["length"]]
            )
        else:
            assert len(rectangles) == 0
            p = segments[0] - np.array(r["center"])
            u = np.cos(r["angle"]) * p[:, 0] - np.sin(r["angle"]) * p[:, 1]
            v = np.sin(r["angle"]) * p[:, 0] + np.cos(r["angle"]) * p[:, 1]
            np.testing.assert_allclose(u, 0, atol=2e-5)
            np.testing.assert_allclose(v, [-r["length"] / 2 + 1, r["length"] / 2 - 1], atol=2e-5)
            np.testing.assert_array_equal(widths, 2)


def test_endpoint_quantization_is_audited_without_hiding_interior_errors():
    import pytest
    from attraction_graph import decode, logits_from_target, target
    from attraction_width_data import audit_roundtrip
    from centerline_grid import render

    s, w, rect = geometry(record(25))
    t, _ = target(s + 32, w, rect + 32, 320, base_radius=1)
    image, stats = decode(logits_from_target(t, base_radius=1), base_radius=1)
    reference = render(s, w, rect, 256)
    audit = audit_roundtrip(t, s, image, reference, stats)
    assert not audit["strict"]
    assert 0.015 < audit["endpoint_shortfall"] < 0.017
    damaged = image.copy()
    damaged[160, 105, 0] = 1
    with pytest.raises(AssertionError):
        audit_roundtrip(t, s, damaged, reference, stats)
