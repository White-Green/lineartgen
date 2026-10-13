import jax
import jax.numpy as jnp
import numpy as np
from cue_capacity import make_arm
from family_objective import family_costs
from local_recursive_fit import fresh_model
from stable_assignment_objective import assignment_gradient, assignment_loss
from stable_assignment_study import update
from test_family_objective import candidates
from test_plausible_lineart import fixture

from lineartgen.diffusion import pyramid
from lineartgen.training import checkpoints, lineart_state


def test_stable_is_one_coherent_answer_not_an_average_or_dynamic_minimum():
    a, b, family = candidates()
    t = pyramid(jnp.full_like(a, 0.5), 2)
    costs = family_costs(pyramid(a, 2), family, t)
    assert float(assignment_loss(costs, jnp.array([0]), "stable")) == 0
    assert float(assignment_loss(costs, jnp.array([1]), "dynamic")) == 0
    assert float(assignment_loss(costs, jnp.array([1]), "stable")) > 0
    # Candidate ordering changes only its id, not loss. Denominator still uses all candidates.
    np.testing.assert_array_equal(
        assignment_loss(costs, jnp.array([1]), "stable"),
        assignment_loss(costs[:, ::-1], jnp.array([0]), "stable"),
    )
    tt = (t[0].at[:, :, :16].set(0),)
    grad = jax.grad(
        lambda v: assignment_loss(family_costs((v,), family, tt), jnp.array([1]), "stable")
    )((a + b) / 2)
    assert np.all(np.asarray(grad)[:, :, :16] == 0)


def test_actual_updates_teacher_unused_and_full_restore(tmp_path):
    config, p, _ = fixture()
    _, _, family = candidates()
    choice = jnp.array([1])
    arms = {}
    for method in ("dynamic", "stable"):
        arm = make_arm(config, fresh_model(config))
        arm["gradient"] = assignment_gradient(method)
        altered = p._replace(targets=tuple(-v for v in p.targets))
        jax.tree.map(
            np.testing.assert_array_equal,
            arm["gradient"](arm["model"], p, family, choice),
            arm["gradient"](arm["model"], altered, family, choice),
        )
        assert update(arm, p, family, choice)["gradient_norm"] > 0
        arms[method] = arm
    with checkpoints.manager(tmp_path / "saved") as saved:
        lineart_state.save(saved, arms, 1)
        restored = {m: make_arm(config, fresh_model(config)) for m in arms}
        for m, arm in restored.items():
            arm["gradient"] = assignment_gradient(m)
        assert lineart_state.restore(saved, restored) == 1
    for m in arms:
        assert update(arms[m], p, family, choice) == update(restored[m], p, family, choice)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 2),
        lineart_state.payload(restored, 2),
    )
