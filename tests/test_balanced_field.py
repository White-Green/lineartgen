import jax
import jax.numpy as jnp
import numpy as np
import test_subpixel_field as base_tests
from balanced_field_objective import balanced_gradient, field_parts
from subpixel_field_data import candidate_fields
from subpixel_field_objective import field_parts as original_parts

from lineartgen.diffusion import pyramid


def test_true_geometry_zero_and_zero_weight_matches_original():
    _, base = base_tests.fixture()
    targets = candidate_fields(base, jnp.array([[0.0, 1.0]]), 3)
    fields = tuple(jnp.clip(t[:, 4], -8, 8) for t in targets)
    strengths = pyramid(jnp.full((1, 32, 32, 1), 0.5), 3)
    result = field_parts(fields, targets, strengths)
    assert float(result["loss"]) < 1e-7 and float(result["sign_margin"]) < 1e-7
    wrong = tuple(f + 0.4 for f in fields)
    a = original_parts(wrong, targets, strengths)
    b = field_parts(wrong, targets, strengths, sign_weight=0)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    ga = jax.grad(lambda f: original_parts(f, targets, strengths)["loss"])(wrong)
    gb = jax.grad(lambda f: field_parts(f, targets, strengths, sign_weight=0)["loss"])(wrong)
    jax.tree.map(np.testing.assert_array_equal, ga, gb)


def test_balanced_inside_outside_gradients_and_protection():
    # A thin negative core amid a much larger positive neighborhood.
    target = jnp.ones((1, 1, 16, 16, 1)).at[:, :, 7:9].set(-0.5)
    strength = jnp.ones((1, 4, 4, 1))
    pred = jnp.zeros((1, 16, 16, 1))

    def sign(f):
        return field_parts((f,), (target,), (strength,))["sign_margin"]

    g = jax.grad(sign)(pred)
    assert np.all(np.asarray(g[:, 7:9]) > 0)
    assert np.all(np.asarray(g[:, :7]) < 0)
    np.testing.assert_allclose(g[:, 7:9].sum(), 0.5, atol=1e-7)
    np.testing.assert_allclose(np.asarray(g, np.float64).sum(), 0, atol=1e-7)
    # Target margin is finite: true signed geometry incurs zero cost.
    assert float(sign(target[:, 0])) == 0
    protected = strength.at[:, :, :2].set(0)
    grad = jax.grad(lambda f: field_parts((f,), (target,), (protected,))["loss"])(pred)
    np.testing.assert_array_equal(grad[:, :, :8], 0)


def test_balanced_actual_update_restore_export_and_six_stages(tmp_path, monkeypatch):
    monkeypatch.setattr(base_tests, "field_gradient", balanced_gradient)
    base_tests.test_real_update_restore_export_and_recursion(tmp_path)
