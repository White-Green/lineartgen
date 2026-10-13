import jax
import jax.numpy as jnp
import numpy as np
from cue_capacity import make_arm
from interval_family_objective import interval_costs, interval_gradient, interval_penalty
from local_recursive_fit import fresh_model
from path_profile_objective import profile_gradient
from robust_family_objective import robust_costs
from stable_assignment_study import update
from test_family_objective import candidates
from test_path_profile import annotation
from test_plausible_lineart import fixture

from lineartgen.diffusion import pyramid


def test_interval_values_and_physical_inward_gradients():
    target = jnp.array([0, 0, 0.3, 0.3, 1, 1.0])
    physical = jnp.array([2, -2, 2, -2, 2, -2.0])

    def loss(z):
        return interval_penalty((1 - z) / 2, target).sum()

    grad = np.asarray(jax.grad(loss)(physical))
    assert grad[0] == grad[5] == 0
    assert grad[1] < 0 and grad[2] > 0 and grad[3] < 0 and grad[4] > 0
    inside = jnp.linspace(0, 1, 11)[:, None]
    np.testing.assert_allclose(
        interval_penalty(inside, target),
        jnp.sqrt((inside - target) ** 2 + 0.05**2) - 0.05,
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_array_equal(
        jax.grad(lambda x: interval_penalty(x, target).sum())(target), jnp.zeros_like(target)
    )


def test_valid_family_equivalence_and_masked_wrong_saturation_gradient():
    a, b, v = candidates()
    t = pyramid(jnp.full_like(a, 0.5), 2)
    for pred in (a, b, (a + b) / 2):
        np.testing.assert_allclose(
            interval_costs(pyramid(pred, 2), v, t), robust_costs(pyramid(pred, 2), v, t)
        )
    raw = jnp.full_like(a, 2.0)
    mask = t[0].at[:, :, :16].set(0)
    grad = jax.grad(lambda z: interval_costs((z,), v[:, 1:2], (mask,)).sum())(raw)
    assert np.all(np.asarray(grad)[:, :, :16] == 0)
    assert np.all(np.asarray(grad)[0, 4:28, 20:22, 0] > 0)
    clipped = jax.grad(lambda z: robust_costs((jnp.clip(z, -1, 1),), v[:, 1:2], (mask,)).sum())(raw)
    assert np.all(np.asarray(clipped) == 0)


def test_actual_update_and_unchanged_control():
    config, p, _ = fixture()
    _, _, v = candidates()
    ann = annotation()
    model = fresh_model(config)
    jax.tree.map(
        np.testing.assert_array_equal,
        interval_gradient("clipped")(model, p, v, ann),
        profile_gradient(0.25)(model, p, v, ann),
    )
    for method in ("clipped", "interval"):
        arm = make_arm(config, model)
        arm["gradient"] = interval_gradient(method)
        assert update(arm, p, v, ann)["gradient_norm"] > 0
