import numpy as np
from attraction_graph import logits_from_target, target
from projected_edge_gate import gate


def test_preserve_true_edges_all_stages():
    segments = np.array([[[8, 8], [45, 45]], [[8, 11], [45, 48]], [[20, 8], [20, 52]]], float)
    widths = np.ones((3, 2)) * 2
    for level in range(6):
        truth, _ = target(segments, widths, np.empty((0, 4)), 64, level)
        h = logits_from_target(truth, level)
        g, rejected = gate(h, level)
        assert rejected == 0
        np.testing.assert_array_equal(g[..., 4:8] >= 0, h[..., 4:8] >= 0)


def test_cross_path_bridge_removed_preserving_nodes_and_widths():
    h = np.full((8, 8, 9), -20, np.float32)
    h[..., 1:4] = 0
    h[3, 3, 0] = h[3, 4, 0] = 20
    h[3, 3, 4] = 20
    h[3, 3, 1] = -1
    h[3, 4, 1] = 1
    g, n = gate(h)
    assert n == 1 and g[3, 3, 4] < 0
    np.testing.assert_array_equal(g[..., :4], h[..., :4])
    np.testing.assert_array_equal(g[..., 8], h[..., 8])
    again, n = gate(g)
    assert n == 0
    np.testing.assert_array_equal(g, again)
