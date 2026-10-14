import jax
import jax.numpy as jnp
import numpy as np
import tail_sign_geometry_study as study
import test_subpixel_field as base_tests
from censored_field_objective import field_parts as mean_parts
from rough_family_data import BASE
from tail_field_objective import field_parts, rms_gradient
from test_focused_geometry import data


def test_sparse_sign_errors_receive_more_attention_without_penalizing_correct_pixels():
    target = jnp.full((1, 1, 16, 16, 1), 8.0)
    strength = jnp.ones((1, 4, 4, 1))
    correct = jnp.full((1, 16, 16, 1), 2.0)
    sparse = correct.at[:, 7, 7].set(-0.75)
    dense = jnp.full_like(correct, -0.75)

    def sign(p):
        return field_parts((p,), (target,), (strength,))["sign_margin"]

    def mean(p):
        return mean_parts((p,), (target,), (strength,))["sign_margin"]

    assert float(sign(sparse)) > 10 * float(mean(sparse))
    np.testing.assert_allclose(float(sign(dense)) / float(mean(dense)), 0.99005, atol=1e-5)
    assert float(sign(correct)) == 0
    np.testing.assert_array_equal(jax.grad(sign)(correct), 0)
    g = np.asarray(jax.grad(sign)(sparse))
    assert g[0, 7, 7, 0] < 0 and np.isfinite(g).all()
    assert np.count_nonzero(g) == 1
    # Distance and derivative terms are identical on this one-candidate fixture.
    a, b = (
        field_parts((sparse,), (target,), (strength,)),
        mean_parts((sparse,), (target,), (strength,)),
    )
    for key in ("distance", "geometry_gradient"):
        np.testing.assert_array_equal(a[key], b[key])


def test_inside_and_outside_wrong_signs_are_corrected_and_protected():
    target = jnp.ones((1, 1, 16, 16, 1)).at[:, :, 7:9].set(-0.5)
    p = jnp.zeros((1, 16, 16, 1))
    strength = jnp.ones((1, 4, 4, 1))
    g = jax.grad(lambda x: field_parts((x,), (target,), (strength,))["sign_margin"])(p)
    assert np.all(np.asarray(g[:, 7:9]) > 0) and np.all(np.asarray(g[:, :7]) < 0)
    protected = strength.at[:, :, :2].set(0)
    g = jax.grad(lambda x: field_parts((x,), (target,), (protected,))["loss"])(p)
    np.testing.assert_array_equal(g[:, :, :8], 0)
    assert float(field_parts((target[:, 0],), (target,), (strength,))["loss"]) < 1e-7


def test_both_arms_share_actual_rough_inputs(monkeypatch):
    opts = BASE | dict(opacity=0.2, sigma=2.4, amplitude=1.5)
    monkeypatch.setattr(
        study, "select", lambda step: dict(indices=[0] * 4, depth=3, augmentation=[opts] * 4)
    )
    arrays = data()
    a, b = study.batch(arrays, 0, "mean"), study.batch(arrays, 0, "rms")
    jax.tree.map(np.testing.assert_array_equal, a, b)
    assert np.max(abs(np.asarray(a[0].guides[0]) - arrays["source"])) > 0.05


def test_rms_real_update_restore_export_and_six_stages(tmp_path, monkeypatch):
    monkeypatch.setattr(base_tests, "field_gradient", rms_gradient)
    base_tests.test_real_update_restore_export_and_recursion(tmp_path)
