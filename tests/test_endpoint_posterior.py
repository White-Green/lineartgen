import numpy as np
import pytest
from endpoint_posterior import posterior
from scipy.special import expit


def values():
    a = np.array([-1.0, 1.0])[None, None, :, None]
    b = np.ones_like(a)
    s = np.full_like(a, 0.4)
    t = np.full_like(a, 0.5)
    x = np.array([-0.1, 0.7])[None, None, :, None]
    return x, t, s, a, b


def test_equal_prior_at_initial_time_and_near_endpoint_teacher_limit():
    x, t, s, a, b = values()
    p, mean, odds = posterior((x,), (t,), (s,), (a,), (b,), 0)
    np.testing.assert_array_equal(p, 0.5)
    np.testing.assert_array_equal(odds, 0)
    np.testing.assert_array_equal(mean, (a + b) / 2)
    for teacher, wanted in ((a, 1), (b, 0)):
        tau = 0.95
        p, mean, _ = posterior(((1 - tau) * s + tau * teacher,), (t,), (s,), (a,), (b,), tau)
        np.testing.assert_array_equal(p, wanted)
        np.testing.assert_array_equal(mean, teacher)


def test_matches_direct_gaussian_density_ratio_and_sums_scale_evidence():
    x, t, s, a, b = values()
    tau = 0.05
    mu_a = (1 - tau) * s + tau * a
    mu_b = (1 - tau) * s + tau * b
    variance = ((1 - tau) * 0.25 * t) ** 2
    expected = np.sum(((x - mu_b) ** 2 - (x - mu_a) ** 2) / (2 * variance))
    p, _, odds = posterior((x,), (t,), (s,), (a,), (b,), tau)
    np.testing.assert_allclose(odds, expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(p, expit(expected), atol=1e-15, rtol=0)
    p, _, odds = posterior((x, x), (t, t), (s, s), (a, a), (b, b), tau)
    np.testing.assert_allclose(odds, 2 * expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(p, expit(2 * expected), atol=1e-15, rtol=0)


def test_protected_evidence_is_ignored_and_invalid_times_rejected():
    x, t, s, a, b = values()
    t[:, :, :1] = 0
    p, _, odds = posterior((x,), (t,), (s,), (a,), (b,), 0.5)
    np.testing.assert_array_equal(p, 0.5)
    np.testing.assert_array_equal(odds, 0)
    x[:, :, :1] = 1e20
    p, _, _ = posterior((x,), (t,), (s,), (a,), (b,), 0.5)
    np.testing.assert_array_equal(p, 0.5)
    for tau in (-0.1, 1, 2):
        with pytest.raises(ValueError):
            posterior((x,), (t,), (s,), (a,), (b,), tau)
