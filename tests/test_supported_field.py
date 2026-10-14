import jax
import jax.numpy as jnp
import numpy as np
import test_subpixel_field as base_tests
from balanced_field_objective import field_parts as baseline_parts
from subpixel_field_data import candidate_fields
from supported_field_objective import field_parts, source_supports, supported_gradient

from lineartgen.diffusion import ProgressTrainingInputs, pyramid


def test_zero_support_matches_balanced_and_true_geometry_zero():
    _, base = base_tests.fixture()
    targets = candidate_fields(base, jnp.array([[0.0, 1.0]]), 3)
    fields = tuple(jnp.clip(t[:, 4], -8, 8) for t in targets)
    strengths = pyramid(jnp.full((1, 32, 32, 1), 0.5), 3)
    supports = tuple(jnp.zeros_like(t) for t in strengths)
    assert float(field_parts(fields, targets, strengths, strengths)["loss"]) < 1e-7
    wrong = tuple(f + 0.4 for f in fields)
    a = baseline_parts(wrong, targets, strengths)
    b = field_parts(wrong, targets, strengths, supports)
    jax.tree.map(np.testing.assert_array_equal, a, b)
    ga = jax.grad(lambda f: baseline_parts(f, targets, strengths)["loss"])(wrong)
    gb = jax.grad(lambda f: field_parts(f, targets, strengths, supports)["loss"])(wrong)
    jax.tree.map(np.testing.assert_array_equal, ga, gb)


def test_support_equalizes_exterior_gradients_without_changing_interior():
    target = jnp.full((1, 1, 16, 16, 1), 8.0)
    target = target.at[:, :, 6:10].set(1.0).at[:, :, 7:9].set(-0.5)
    strength = jnp.ones((1, 4, 4, 1))
    pred = jnp.zeros((1, 16, 16, 1))

    def grad(support, active=strength):
        return jax.grad(
            lambda f: field_parts((f,), (target,), (active,), (support,))["sign_margin"]
        )(pred)

    baseline = grad(jnp.zeros_like(strength))
    supported = grad(strength)
    np.testing.assert_array_equal(baseline[:, 7:9], supported[:, 7:9])
    np.testing.assert_allclose(baseline[0, 0, 0] / baseline[0, 6, 0], 0.01, rtol=1e-6)
    np.testing.assert_array_equal(supported[0, 0, 0], supported[0, 6, 0])
    assert np.all(np.asarray(supported[:, :7]) < 0)
    np.testing.assert_allclose(np.asarray(supported, np.float64).sum(), 0, atol=1e-7)
    protected = grad(strength, strength.at[:, :, :2].set(0))
    np.testing.assert_array_equal(protected[:, :, :8], 0)


def test_source_supports_are_input_only_and_area_preserving():
    image = jnp.ones((1, 32, 32, 1)).at[:, 12:20, 8:24].set(0.8)
    guides = pyramid(image, 3)
    prepared = ProgressTrainingInputs(guides, guides, guides, guides, guides)
    supports = source_supports(prepared)
    assert float(supports[0].min()) == 0
    np.testing.assert_allclose(supports[0][:, 12:20, 8:24], 1, atol=1e-6)
    for i, support in enumerate(supports):
        np.testing.assert_allclose(support.sum() * 4**i, supports[0].sum(), rtol=1e-6)
    zero = tuple(jnp.zeros_like(x) for x in guides)
    changed = source_supports(prepared._replace(noisy=zero, targets=zero))
    jax.tree.map(np.testing.assert_array_equal, supports, changed)


def test_supported_actual_update_restore_export_and_six_stages(tmp_path, monkeypatch):
    monkeypatch.setattr(base_tests, "field_gradient", supported_gradient)
    base_tests.test_real_update_restore_export_and_recursion(tmp_path)
