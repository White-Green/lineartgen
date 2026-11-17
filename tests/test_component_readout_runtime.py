"""Readout assets and whole-path decisions must preserve the graph contract."""

import numpy as np
import pytest
from lineartgen_runtime.component_readout import SHAPES, descriptors, load_parameters, project
from lineartgen_runtime.selection_feedback import project as original_project


def fixture():
    base = np.full((7, 9, 9), -20.0, np.float32)
    base[..., 1:4] = 0
    for row in (1, 5):
        base[row, 1:8, 0] = 2
        base[row, 1:7, 4] = 2
    head = base.copy()
    head[..., 0] = -0.5
    state = np.zeros((7, 9, 16), np.float32)
    state[1, 1:8, 0] = 2
    state[5, 1:8, 0] = -2
    params = {k: np.zeros(shape, np.float32) for k, shape in SHAPES.items()}
    params["hidden_kernel"][0, 0] = 1
    params["kernel"][0, 0] = 1
    return base, head, state, params


def test_readout_keeps_complete_path_without_changing_width_position_or_edges():
    base, head, state, params = fixture()
    result = project(head, state, base, params)
    np.testing.assert_array_equal(result[1, 1:8, 0], base[1, 1:8, 0])
    assert np.all(result[5, 1:8, 0] < 0)
    np.testing.assert_array_equal(result[..., 1:], base[..., 1:])
    assert not np.any((result[..., 0] >= 0) & (base[..., 0] < 0))
    np.testing.assert_array_equal(base, fixture()[0])
    # Features on another component or inactive cells cannot affect this path's readout.
    state[5, 1:8, :] = 100
    state[3, :, :] = 1000
    changed = project(head, state, base, params)
    np.testing.assert_array_equal(changed[1, 1:8], result[1, 1:8])


def test_pooling_retains_mean_and_max_and_ignores_unoccupied_cells():
    ids = np.array([[0, 0, -1], [1, 1, 1]], np.int32)
    state = np.zeros((2, 3, 16), np.float32)
    state[..., 0] = [[-2, 4, 1000], [1, 2, 6]]
    features, mass = descriptors(state, ids)
    np.testing.assert_array_equal(mass, [2, 3])
    np.testing.assert_allclose(features[:, 0], [1, 3])
    np.testing.assert_array_equal(features[:, 16], [4, 6])
    assert not features[:, 1:16].any() and not features[:, 17:].any()


def test_zero_readout_preserves_original_projection_including_zero_and_empty_scores():
    base, head, state, params = fixture()
    params["kernel"][:] = 0
    head[1, :, 0] = 0
    np.testing.assert_array_equal(project(head, state, base, params), original_project(head, base))
    base[..., 0] = -20
    np.testing.assert_array_equal(project(head, state, base, params), base)


@pytest.mark.parametrize("fault", ["missing", "extra", "shape", "dtype", "nan"])
def test_invalid_parameter_archives_are_rejected(tmp_path, fault):
    params = fixture()[3]
    if fault == "missing":
        del params["bias"]
    elif fault == "extra":
        params["unused"] = np.zeros(1, np.float32)
    elif fault == "shape":
        params["kernel"] = np.zeros((31, 1), np.float32)
    elif fault == "dtype":
        params["kernel"] = params["kernel"].astype(np.float64)
    else:
        params["bias"][0] = np.nan
    path = tmp_path / "readout.npz"
    np.savez(path, **params)
    with pytest.raises(ValueError, match="Invalid component readout parameter"):
        load_parameters(path)


def test_parameter_archive_is_exact_and_read_only(tmp_path):
    params = fixture()[3]
    path = tmp_path / "readout.npz"
    np.savez(path, **params)
    loaded = load_parameters(path)
    for k in params:
        np.testing.assert_array_equal(loaded[k], params[k])
        assert not loaded[k].flags.writeable
