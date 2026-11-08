"""Selection must keep connected paths intact without merging nearby paths."""

import numpy as np
import pytest
from lineartgen_runtime.selection_feedback import components, project


def two_paths():
    base = np.full((7, 9, 9), -20.0, np.float32)
    base[..., 1:4] = 0
    for row in (1, 5):
        base[row, 1:8, 0] = 2
        base[row, 1:7, 4] = 2
    return base


def test_whole_path_decision_restores_local_cut_and_preserves_geometry():
    base = two_paths()
    head = base.copy()
    head[1, 1:8, 0] = 3
    head[1, 4, 0] = -4  # A local CNN mistake must not cut the kept path.
    head[5, 1:8, 0] = -3
    head[5, 4, 0] = 4  # An isolated positive score must not keep the rejected path.
    result = project(head, base)
    np.testing.assert_array_equal(result[1, 1:8, 0], base[1, 1:8, 0])
    assert np.all(result[5, 1:8, 0] < 0)
    np.testing.assert_array_equal(result[..., 1:], base[..., 1:])
    np.testing.assert_array_equal(base, two_paths())


def test_component_connections_follow_edges_and_rendered_distance():
    base = two_paths()
    base[1, 4, 4] = -20  # Deliberately open edge.
    ids = components(base)
    assert ids[1, 1] == ids[1, 4]
    assert ids[1, 4] != ids[1, 5]
    assert ids[1, 1] != ids[5, 1]
    base[1, 4, 4] = 2
    base[1, 4, 1] = -10
    base[1, 5, 1] = 10  # Neighboring cells but points farther than the renderer permits.
    ids = components(base)
    assert ids[1, 4] != ids[1, 5]


def test_empty_and_all_kept_graphs_are_unchanged():
    base = two_paths()
    np.testing.assert_array_equal(project(base, base), base)
    base[..., 0] = -20
    np.testing.assert_array_equal(project(base, base), base)
    assert np.all(components(base) == -1)


def test_invalid_scores_fail_before_projection():
    base = two_paths()
    with pytest.raises(ValueError):
        project(base, base, threshold=float("nan"))
    base[1, 1, 0] = np.nan
    with pytest.raises(ValueError):
        project(base, base)
