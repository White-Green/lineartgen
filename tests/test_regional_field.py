import jax
import jax.numpy as jnp
import numpy as np
import region_sign_geometry_study as study
import test_subpixel_field as base_tests
from regional_field_objective import field_parts, regional_gradient
from rough_family_data import BASE
from tail_field_objective import field_parts as shared_parts
from test_focused_geometry import data


def test_equivalent_near_far_errors_receive_equal_sign_gradients():
    target = jnp.full((1, 1, 16, 16, 1), -8.0).at[:, :, :8].set(-0.5)
    strength = jnp.ones((1, 4, 4, 1))
    pred = jnp.full((1, 16, 16, 1), -2.0).at[:, 3, 7].set(1).at[:, 12, 7].set(1)
    old = jax.grad(lambda x: shared_parts((x,), (target,), (strength,))["sign_margin"])(pred)
    new = jax.grad(lambda x: field_parts((x,), (target,), (strength,))["sign_margin"])(pred)
    np.testing.assert_allclose(float(old[0, 12, 7, 0] / old[0, 3, 7, 0]), 0.01, atol=1e-7)
    np.testing.assert_array_equal(new[0, 12, 7, 0], new[0, 3, 7, 0])
    assert float(new[0, 12, 7, 0]) > float(old[0, 12, 7, 0]) * 10
    correct = jnp.full_like(pred, -2.0)

    def sign(x):
        return field_parts((x,), (target,), (strength,))["sign_margin"]

    assert float(sign(correct)) == 0
    np.testing.assert_array_equal(jax.grad(sign)(correct), 0)
    protected = strength.at[:, :, :2].set(0)
    g = jax.grad(lambda x: field_parts((x,), (target,), (protected,))["loss"])(pred)
    np.testing.assert_array_equal(g[:, :, :8], 0)


def test_all_near_matches_shared_rms_and_empty_regions_are_safe():
    target = jnp.ones((1, 1, 16, 16, 1)).at[:, :, 7:9].set(-0.5)
    pred = jnp.zeros((1, 16, 16, 1))
    strength = jnp.ones((1, 4, 4, 1))
    a, b = (
        field_parts((pred,), (target,), (strength,)),
        shared_parts((pred,), (target,), (strength,)),
    )
    jax.tree.map(np.testing.assert_array_equal, a, b)
    a = jax.grad(lambda x: field_parts((x,), (target,), (strength,))["loss"])(pred)
    b = jax.grad(lambda x: shared_parts((x,), (target,), (strength,))["loss"])(pred)
    np.testing.assert_array_equal(a, b)


def test_both_arms_share_rough_batch(monkeypatch):
    opts = BASE | dict(opacity=0.2, sigma=2.4, amplitude=1.5)
    monkeypatch.setattr(
        study, "select", lambda step: dict(indices=[0] * 4, depth=3, augmentation=[opts] * 4)
    )
    arrays = data()
    jax.tree.map(
        np.testing.assert_array_equal,
        study.batch(arrays, 0, "control"),
        study.batch(arrays, 0, "regions"),
    )


def test_regional_real_update_restore_export_and_six_stages(tmp_path, monkeypatch):
    monkeypatch.setattr(base_tests, "field_gradient", regional_gradient)
    base_tests.test_real_update_restore_export_and_recursion(tmp_path)
