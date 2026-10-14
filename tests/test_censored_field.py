import censored_geometry_study as study
import jax
import jax.numpy as jnp
import numpy as np
import test_subpixel_field as base_tests
from censored_field_objective import censored_gradient, field_parts
from subpixel_field_data import candidate_fields
from test_focused_geometry import data

from lineartgen.diffusion import pyramid


def test_exact_geometry_and_saturated_regions_are_free():
    _, base = base_tests.fixture()
    targets = candidate_fields(base, jnp.array([[0.0, 1.0]]), 3)
    fields = tuple(jnp.clip(t[:, 4], -8, 8) for t in targets)
    strengths = pyramid(jnp.full((1, 32, 32, 1), 0.5), 3)
    assert float(field_parts(fields, targets, strengths)["loss"]) < 1e-7
    for sign in (-1, 1):
        target = jnp.full((1, 1, 16, 16, 1), sign * 8.0)
        pred = jnp.full((1, 16, 16, 1), sign * 2.0)
        strength = jnp.ones((1, 4, 4, 1))

        def fn(p, target=target, strength=strength):
            return field_parts((p,), (target,), (strength,))["loss"]

        assert float(fn(pred)) == 0
        np.testing.assert_array_equal(jax.grad(fn)(pred), 0)
        wrong = -pred * 10
        g = jax.grad(fn)(wrong)
        assert float(fn(wrong)) > 0
        assert np.all(np.asarray(g) * sign < 0)


def test_boundary_distance_and_gradient_remain_supervised_and_protected():
    target = jnp.broadcast_to(
        jnp.linspace(-0.75, 0.75, 16)[None, None, :, None, None], (1, 1, 16, 16, 1)
    )
    pred = jnp.zeros((1, 16, 16, 1))
    strength = jnp.ones((1, 4, 4, 1))
    parts = field_parts((pred,), (target,), (strength,))
    assert float(parts["distance"]) > 0 and float(parts["geometry_gradient"]) > 0
    g = jax.grad(lambda p: field_parts((p,), (target,), (strength,))["loss"])(pred)
    assert np.linalg.norm(g) > 0
    protected = strength.at[:, :, :2].set(0)
    g = jax.grad(lambda p: field_parts((p,), (target,), (protected,))["loss"])(pred)
    np.testing.assert_array_equal(g[:, :, :8], 0)


def test_control_and_censored_receive_identical_prepared_batches(monkeypatch):
    monkeypatch.setattr(study, "select", lambda step: dict(indices=[0] * 4, depth=3))
    arrays = data()
    jax.tree.map(
        np.testing.assert_array_equal,
        study.batch(arrays, 0, "control"),
        study.batch(arrays, 0, "censored"),
    )


def test_censored_real_update_restore_export_and_six_stages(tmp_path, monkeypatch):
    monkeypatch.setattr(base_tests, "field_gradient", censored_gradient)
    base_tests.test_real_update_restore_export_and_recursion(tmp_path)
