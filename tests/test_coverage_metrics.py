import numpy as np
import pytest

from lineartgen.coverage_metrics import coverage_metrics


def image(alpha):
    return (1 - 2 * np.asarray(alpha, dtype=np.float64)).reshape(1, -1, 1)


def test_fractional_identity_and_known_mass_loss():
    target = image([0, 0.1, 0.5, 0.9, 1])
    strength = np.ones_like(target)
    assert coverage_metrics(target, target, strength)["coverage_overlap_f1"] == 1
    result = coverage_metrics(image([0, 0.05, 0.25, 0.45, 0.5]), target, strength)
    assert result["coverage_overlap_f1"] == pytest.approx(2 / 3)


def test_symmetry_bounds_and_l1_identity():
    rng = np.random.default_rng(5)
    a, b = rng.uniform(-1, 1, (2, 16, 16, 1))
    strength = np.ones_like(a)
    x = coverage_metrics(a, b, strength)
    y = coverage_metrics(b, a, strength)
    assert x == y and 0 <= x["coverage_overlap_f1"] <= 1
    assert x["coverage_overlap_f1"] == pytest.approx(
        1 - x["active_alpha_l1_sum"] / x["total_predicted_and_target_coverage"]
    )


def test_empty_active_paper_black_and_protected_region():
    target = image([0, 0, 1, 1])
    predicted = image([1, 1, 1, 1])
    assert coverage_metrics(predicted, target, np.zeros_like(target))["coverage_overlap_f1"] == 1
    assert coverage_metrics(image([0]), image([0]), image([0]))["coverage_overlap_f1"] == 1
    assert coverage_metrics(image([1]), image([0]), np.ones((1, 1, 1)))["coverage_overlap_f1"] == 0
    mask = np.array([0, 0, 1, 1]).reshape(1, 4, 1)
    assert coverage_metrics(predicted, target, mask)["coverage_overlap_f1"] == 1


def test_near_threshold_change_has_bounded_continuous_cost():
    a, b = image([0.499, 0.499]), image([0.501, 0.501])
    assert coverage_metrics(a, b, np.ones_like(a))["coverage_overlap_f1"] == pytest.approx(0.998)
