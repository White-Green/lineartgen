import numpy as np
from attraction_graph import logits_from_target
from graph_residual_audit import MODES, replace


def test_replacement_only_changes_declared_channels_without_mutation():
    rng = np.random.default_rng(93)
    h = rng.normal(size=(12, 12, 9)).astype(np.float32)
    t = np.zeros_like(h)
    t[2:5, 3:7, 0] = 1
    t[..., 1:3] = 0.2
    t[..., 3] = np.log(2)
    before_h, before_t = h.copy(), t.copy()
    for mode, channels in MODES.items():
        got = replace(h, t, mode)
        for k in set(range(9)) - set(channels):
            np.testing.assert_array_equal(got[..., k], h[..., k])
        if mode in ("GT position at GT vertices", "GT width at GT vertices"):
            np.testing.assert_array_equal(got[t[..., 0] == 0], h[t[..., 0] == 0])
    np.testing.assert_array_equal(h, before_h)
    np.testing.assert_array_equal(t, before_t)


def test_full_oracle_replaces_all_and_graph_preserves_fill():
    rng = np.random.default_rng(94)
    h = rng.normal(size=(8, 8, 9)).astype(np.float32)
    t = np.zeros_like(h)
    t[1:6, 4, 0] = 1
    t[..., 1:3] = 0.15
    t[..., 8] = 1
    oracle = logits_from_target(t, base_radius=1.0)
    np.testing.assert_array_equal(replace(h, t, "full oracle"), oracle)
    graph = replace(h, t, "GT graph")
    np.testing.assert_array_equal(graph[..., :8], oracle[..., :8])
    np.testing.assert_array_equal(graph[..., 8], h[..., 8])
