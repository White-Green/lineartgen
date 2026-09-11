import numpy as np
import pytest
from scipy import ndimage

from lineartgen.evaluation import KINDS, line_case, probe_metrics, protected_metrics


@pytest.mark.parametrize("kind", KINDS)
def test_known_paths_are_calibrated_against_their_rasterized_targets(kind):
    for variant in range(8):
        gray, probes, connections = line_case(kind, variant)
        target = np.clip((gray * 2 - 1) * 1.1, -1, 1)
        metrics = probe_metrics(target, target, probes, connections)
        assert metrics["profile_width_mae_px"] == 0
        assert metrics["profile_missing_fraction"] == 0
        assert metrics["profile_offset_mae_px"] == 0
        assert metrics["connection_cut_rate"] in (None, 0)
        assert metrics["connection_bridge_rate"] in (None, 0)
        assert metrics["connection_queries_valid"] > 0


def test_probes_distinguish_cuts_bridges_width_and_position_changes():
    gray, probes, connections = line_case("straight", 0)
    target = np.clip((gray * 2 - 1) * 1.1, -1, 1)
    cut = target.copy()
    cut[:, 252:260] = 1
    assert probe_metrics(cut, target, probes, connections)["connection_cut_rate"] == 1
    missing = np.ones_like(target)
    assert probe_metrics(missing, target, probes, connections)["profile_missing_fraction"] == 1
    assert probe_metrics(missing, target, probes, connections)["profile_offset_mae_px"] is None
    widened = ndimage.minimum_filter(target, size=(5, 1, 1))
    assert probe_metrics(widened, target, probes, connections)["profile_width_mae_px"] > 3
    shifted = ndimage.shift(target, (3, 0, 0), order=1, mode="constant", cval=1)
    assert probe_metrics(shifted, target, probes, connections)["profile_offset_mae_px"] > 2.9
    gray, probes, connections = line_case("gap", 0)
    target = np.clip((gray * 2 - 1) * 1.1, -1, 1)
    bridge = target.copy()
    x, y = np.rint(connections[-1, :4].reshape(2, 2).mean(axis=0)).astype(int)
    bridge[y - 1 : y + 2, x - 10 : x + 11] = -1
    assert probe_metrics(bridge, target, probes, connections)["connection_bridge_rate"] == 1


def test_protected_maximum_and_mask_boundary_do_not_hide_local_damage():
    target = np.ones((32, 32, 1), np.float32)
    noise = np.ones_like(target)
    noise[:, :16] = 0
    changed = target.copy()
    changed[16, 15] = -1
    metrics = protected_metrics(changed, target, noise)
    assert metrics["protected_max_error"] == 2
    assert metrics["mask_boundary_mae"] > 0
