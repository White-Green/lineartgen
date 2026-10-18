import numpy as np
from attraction_graph import decode, logits_from_target, radius, target
from centerline_grid import render


def encoded(lines, widths=None, ids=None, level=0):
    s = np.asarray(lines, np.float64).reshape(-1, 2, 2)
    w = np.full((len(s), 2), 2.0) if widths is None else np.asarray(widths)
    t, stats = target(s, w, np.empty((0, 4)), 32, level, ids)
    image, graph = decode(logits_from_target(t, level), level)
    return t, stats, image, graph


def test_anchors_project_to_line_instead_of_widening_it():
    s = [[[4.2, 16.2], [27.7, 16.2]]]
    t, stats, image, g = encoded(s)
    y, x = np.nonzero(t[..., 0])
    projected = np.stack((x + 0.5, y + 0.5), -1) + t[y, x, 1:3]
    np.testing.assert_allclose(projected[:, 1], 16.2, atol=1e-6)
    assert stats["anchors"] > 100 and g["components"] == 1
    np.testing.assert_allclose(image, render(s, [[2, 2]], [], 32), atol=1e-5)


def test_close_distinct_paths_are_not_connected():
    s = [[[4.2, 16.1], [27.7, 16.1]], [[4.2, 16.35], [27.7, 16.35]]]
    _, stats, _, g = encoded(s, [[0.2, 0.2], [0.2, 0.2]], [0, 1])
    assert stats["represented_paths"] == 2 and g["components"] == 2


def test_gap_and_short_segment_endpoints_survive():
    _, _, _, g = encoded([[[4.2, 16.2], [16.1, 16.2]], [[16.35, 16.2], [27.7, 16.2]]], ids=[0, 1])
    assert g["components"] == 2
    s = [[[16.1, 16.1], [16.3, 16.1]]]
    _, _, image, g = encoded(s, [[0.2, 0.2]])
    assert g["components"] == 1
    np.testing.assert_allclose(image, render(s, [[0.2, 0.2]], [], 32), atol=1e-5)


def test_all_levels_empty_fill_and_border_are_finite():
    for level in range(6):
        t, stats = target(
            np.empty((0, 2, 2)), np.empty((0, 2)), np.array([[0, 0, 32, 32]]), 32, level
        )
        image, _ = decode(logits_from_target(t, level), level)
        assert stats["anchors"] == 0 and np.all(image == -1)
        assert radius(level) >= 0.75
        _, _, image, _ = encoded([[[-0.1, 1], [10, 1]]], level=level)
        assert np.isfinite(image).all()
    _, _, image, _ = encoded([])
    assert np.all(image == 1)


def test_three_subpixel_paths_can_still_exceed_anchor_capacity():
    s = [[[4, y], [28, y]] for y in [16.1, 16.2, 16.3]]
    _, stats, _, _ = encoded(s, ids=[0, 1, 2])
    assert stats["represented_paths"] < 3  # Explicit remaining representation limit.
