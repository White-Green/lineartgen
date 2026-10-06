import numpy as np

from lineartgen.lineart_metrics import halo_metrics


def test_visible_gray_halo_is_detected_despite_perfect_half_threshold_f1():
    truth = np.ones((64, 64, 1), np.float32)
    truth[:, 31:34] = -1
    prediction = truth.copy()
    prediction[:, 27:31] = prediction[:, 34:38] = 0.5
    mask = np.ones_like(truth)
    exact = halo_metrics(truth, truth, mask)
    halo = halo_metrics(prediction, truth, mask)
    missing = halo_metrics(np.ones_like(truth), truth, mask)
    assert halo["ink_f1_alpha50"] == exact["ink_f1_alpha50"] == 1
    assert halo["halo_opacity_excess"] > exact["halo_opacity_excess"] == 0
    assert halo["ink_f1_alpha10"] < exact["ink_f1_alpha10"]
    assert missing["ink_deficit_fraction"] > halo["ink_deficit_fraction"] == 0
    assert halo_metrics(prediction, truth, np.zeros_like(truth))["halo_opacity_excess"] is None
