import jax
import jax.numpy as jnp
import numpy as np
from cue_capacity import make_arm
from linear_family_objective import linear_costs, linear_gradient, linear_penalty
from local_recursive_fit import fresh_model
from path_profile_objective import profile_gradient
from robust_family_objective import robust_costs
from stable_assignment_study import update
from test_family_objective import candidates
from test_path_profile import annotation
from test_plausible_lineart import fixture

from lineartgen.diffusion import pyramid


def test_linear_extension_value_derivative_and_antialias_strength():
    target = jnp.array([0, 0.01, 0.3, 0.5, 0.99, 1.0])
    slope = jax.grad(lambda a: linear_penalty(a, target).sum())
    for bound in (0.0, 1.0):
        residual = bound - target
        expected = residual / jnp.sqrt(residual**2 + 0.05**2)
        for delta in (-1e-7, 0, 1e-7):
            np.testing.assert_allclose(
                slope(jnp.full_like(target, bound + delta)), expected, atol=3e-6
            )
    for alpha in (-2.0, -0.1, 1.1, 3.0):
        bounded = np.clip(alpha, 0, 1)
        expected = (bounded - target) / jnp.sqrt((bounded - target) ** 2 + 0.05**2)
        np.testing.assert_allclose(slope(jnp.full_like(target, alpha)), expected)
    assert slope(jnp.full_like(target, -0.5))[0] == 0
    assert slope(jnp.full_like(target, 1.5))[-1] == 0
    np.testing.assert_allclose(slope(jnp.full_like(target, -0.5))[1], -0.19611613)
    np.testing.assert_array_equal(slope(target), jnp.zeros_like(target))


def test_physical_values_protection_and_actual_updates():
    a, b, v = candidates()
    masks = pyramid(jnp.full_like(a, 0.5), 2)
    for pred in (a, b, (a + b) / 2):
        np.testing.assert_allclose(
            linear_costs(pyramid(pred, 2), v, masks), robust_costs(pyramid(pred, 2), v, masks)
        )
    raw = jnp.full_like(a, 2.0)
    mask = masks[0].at[:, :, :16].set(0)
    grad = jax.grad(lambda z: linear_costs((z,), v[:, 1:2], (mask,)).sum())(raw)
    assert np.all(np.asarray(grad)[:, :, :16] == 0)
    assert np.all(np.asarray(grad)[0, 4:28, 20:22, 0] > 0)
    config, prepared, _ = fixture()
    model = fresh_model(config)
    ann = annotation()
    jax.tree.map(
        np.testing.assert_array_equal,
        linear_gradient("clipped")(model, prepared, v, ann),
        profile_gradient(0.25)(model, prepared, v, ann),
    )
    arm = make_arm(config, model)
    arm["gradient"] = linear_gradient("linear")
    assert update(arm, prepared, v, ann)["gradient_norm"] > 0
