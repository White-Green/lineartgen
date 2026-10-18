import numpy as np
from centerline_grid import encode, pieces, render, segment_distance


def test_straight_endpoints_and_render():
    segments = np.array([[[4.2, 8.2], [25.7, 8.2]]])
    widths = np.ones((1, 2)) * 2
    for cell in (0.5, 1, 2, 4):
        graph = encode(segments, widths, cell)
        assert graph["stats"]["components"] == 1
        np.testing.assert_allclose(graph["vertices"][0], segments[0, 0])
        np.testing.assert_allclose(graph["vertices"][-1], segments[0, 1])
        np.testing.assert_allclose(
            render(segments, widths, [], 32),
            render(graph["segments"], graph["segment_widths"], [], 32),
            atol=1e-6,
        )


def test_corner_crossings_are_adjacent():
    keys = [p[0] for p in pieces([0.25, 0.25], [3.75, 3.75], 1)]
    assert keys == [(0, 0), (1, 1), (2, 2), (3, 3)]
    graph = encode(np.array([[[0.25, 0.25], [3.75, 3.75]]]), np.ones((1, 2)), 1)
    assert graph["stats"]["edges"] == 3


def test_nearby_paths_collide_but_crossing_is_not_false_contact():
    lines = np.array([[[2.1, 4.1], [8.1, 4.1]], [[2.1, 4.6], [8.1, 4.6]]])
    fine, coarse = (encode(lines, np.ones((2, 2)), cell) for cell in (0.5, 1))
    assert fine["stats"]["components"] == 2
    assert fine["stats"]["noncontact_path_collisions"] == 0
    assert coarse["stats"]["components"] == 1
    assert coarse["stats"]["noncontact_path_collisions"] > 0
    assert (
        segment_distance(np.array([[0.0, 0.0], [2.0, 2.0]]), np.array([[0.0, 2.0], [2.0, 0.0]]))
        == 0
    )


def test_blank_dot_fill_and_degenerate_segments():
    empty = encode([], [], 1)
    assert empty["stats"]["vertices"] == 0
    np.testing.assert_array_equal(render([], [], [], 16), 1)
    dot = np.array([[[8.2, 8.2], [8.2, 8.2]]])
    graph = encode(dot, np.ones((1, 2)) * 2, 1)
    assert graph["stats"]["isolated_vertices"] == 1
    np.testing.assert_array_equal(
        render(dot, [[2, 2]], [], 16), render(graph["segments"], graph["segment_widths"], [], 16)
    )
    fill = render([], [], [[2, 2, 14, 14]], 16)
    np.testing.assert_array_equal(fill[3:13, 3:13], -1)


def test_taper_keeps_endpoint_widths_and_short_line_reports_collision():
    graph = encode([[[1.1, 4.2], [15.5, 4.2]]], [[0.1, 3.0]], 1)
    np.testing.assert_allclose(graph["widths"][[0, -1]], [0.1, 3.0])
    short = encode([[[1.1, 1.1], [1.2, 1.1]]], [[1, 1]], 1)
    assert short["stats"]["endpoint_collision_cells"] == 1
    assert short["stats"]["isolated_vertices"] == 1


def test_two_separate_fragments_stay_disconnected_when_cells_separate():
    lines = [[[1.1, 3.2], [5.1, 3.2]], [[7.1, 3.2], [11.1, 3.2]]]
    graph = encode(lines, np.ones((2, 2)), 1)
    assert graph["stats"]["components"] == 2
