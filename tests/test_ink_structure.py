import jax
import jax.numpy as jnp
import numpy as np

from lineartgen.training.ink_structure import ink_structure_losses
from lineartgen.training.lineart_study import METHODS, study_methods


def test_continuous_teacher_is_stationary_including_thin_faint_lines():
    target = jnp.linspace(-1, 1, 17 * 19).reshape(1, 17, 19, 1)
    active = jnp.ones_like(target, dtype=bool)
    for name in ("ink_keep", "neighbor_keep", "neighbor_paper"):
        value, gradient = jax.value_and_grad(
            lambda p, name=name: ink_structure_losses(p, target, active)[name]
        )(target)
        assert float(value) == 0
        np.testing.assert_array_equal(gradient, 0)


def test_missing_ink_and_spurious_bridge_have_opposing_correct_gradients():
    target = jnp.ones((1, 17, 17, 1)).at[:, 7, :, :].set(-0.6).at[:, 9, :, :].set(-0.6)
    prediction = target.at[:, 7, 8, :].set(0.8).at[:, 8, 8, :].set(-0.6)
    active = jnp.ones_like(target, dtype=bool)
    for name in ("ink_keep", "neighbor_keep"):
        gradient = jax.grad(lambda p, name=name: ink_structure_losses(p, target, active)[name])(
            prediction
        )
        assert float(gradient[0, 7, 8, 0]) > 0  # Descent darkens the missing teacher line.
        assert float(gradient[0, 8, 8, 0]) == 0  # Does not reward a false bridge.
    gradient = jax.grad(lambda p: ink_structure_losses(p, target, active)["neighbor_paper"])(
        prediction
    )
    assert float(gradient[0, 8, 8, 0]) < 0  # Descent whitens the gap.
    assert float(gradient[0, 7, 8, 0]) == 0


def test_neighbor_term_emphasizes_a_local_weak_spot_without_changing_teacher():
    target = jnp.ones((1, 17, 17, 1)).at[:, 8, :, :].set(-0.4)
    prediction = target.at[:, 8, 8, :].set(0.5)
    active = jnp.ones_like(target, dtype=bool)
    parts = ink_structure_losses(prediction, target, active)
    assert float(parts["neighbor_keep"]) > float(parts["ink_keep"]) * 2.9
    zero_radius = ink_structure_losses(prediction, target, active, neighbor_radius=0)
    np.testing.assert_array_equal(zero_radius["ink_keep"], zero_radius["neighbor_keep"])


def test_protected_error_has_no_gradient_or_effect_on_editable_neighbors():
    target = jnp.ones((1, 17, 17, 1)).at[:, 8, :, :].set(-0.6)
    active = jnp.ones_like(target, dtype=bool).at[:, :, 8:, :].set(False)
    prediction = target.at[:, :, 8:, :].set(0.0)
    value, gradient = jax.value_and_grad(
        lambda p: sum(ink_structure_losses(p, target, active).values())
    )(prediction)
    assert float(value) == 0
    np.testing.assert_array_equal(gradient, 0)


def test_empty_classes_and_fully_protected_images_are_finite():
    for target in (jnp.ones((2, 8, 8, 1)), -jnp.ones((2, 8, 8, 1))):
        for active in (jnp.zeros_like(target, bool), jnp.ones_like(target, bool)):
            value, gradient = jax.value_and_grad(
                lambda p, target=target, active=active: sum(
                    ink_structure_losses(p, target, active).values()
                )
            )(jnp.zeros_like(target))
            assert np.isfinite(float(value))
            assert np.isfinite(gradient).all()


def test_protocol_methods_keep_legacy_default_and_validate_paths():
    import pytest

    assert study_methods({}) == METHODS
    assert study_methods({"methods": ["control", "ink_guard"]}) == ("control", "ink_guard")
    for names in ([], ["control", "control"], ["../other"], ["x/y"]):
        with pytest.raises(ValueError):
            study_methods({"methods": names})
