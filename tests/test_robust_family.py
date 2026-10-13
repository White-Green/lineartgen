import jax
import jax.numpy as jnp
import numpy as np
from cue_capacity import make_arm
from local_recursive_fit import fresh_model
from robust_family_objective import robust_costs, robust_gradient
from stable_assignment_study import update
from test_family_objective import candidates
from test_plausible_lineart import fixture

from lineartgen.diffusion import pyramid


def test_robust_family_preserves_clean_solutions_and_rejects_gray_hybrid():
    a, b, family = candidates()
    t = pyramid(jnp.full_like(a, 0.5), 2)
    ca = robust_costs(pyramid(a, 2), family, t)
    cb = robust_costs(pyramid(b, 2), family, t)
    assert float(ca.min()) == float(cb.min()) == 0
    assert float(robust_costs(pyramid((a + b) / 2, 2), family, t).min()) > 0
    assert float(robust_costs((a, pyramid(b, 2)[1]), family, t).min()) > 0
    np.testing.assert_array_equal(ca, robust_costs(pyramid(a, 2), family[:, ::-1], t)[:, ::-1])
    protected = t[0].at[:, :, :16].set(0)
    grad = jax.grad(lambda x: robust_costs((x,), family, (protected,)).min())((a + b) / 2)
    assert np.all(np.asarray(grad)[:, :, :16] == 0)
    # No force on exactly matching white pixels, unlike an arbitrary abs(0) subgradient.
    grad = jax.grad(lambda x: robust_costs((x,), family[:, :1], (t[0],)).mean())(a)
    np.testing.assert_array_equal(grad, jnp.zeros_like(a))


def test_robust_actual_update_is_finite_and_teacher_independent():
    config, p, _ = fixture()
    _, _, family = candidates()
    for method in ("dynamic", "stable"):
        arm = make_arm(config, fresh_model(config))
        arm["gradient"] = robust_gradient(method)
        choice = jnp.array([1])
        altered = p._replace(targets=tuple(-v for v in p.targets))
        jax.tree.map(
            np.testing.assert_array_equal,
            arm["gradient"](arm["model"], p, family, choice),
            arm["gradient"](arm["model"], altered, family, choice),
        )
        assert update(arm, p, family, choice)["gradient_norm"] > 0
