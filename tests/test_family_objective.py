import jax
import jax.numpy as jnp
import numpy as np
from cue_capacity import make_arm
from family_objective import family_costs, family_gradient
from family_study import update
from local_recursive_fit import fresh_model
from test_plausible_lineart import fixture

from lineartgen.diffusion import pyramid
from lineartgen.training import checkpoints, lineart_state


def candidates():
    a = jnp.ones((1, 32, 32, 1)).at[:, 4:28, 10:12].set(-1)
    b = jnp.ones_like(a).at[:, 4:28, 20:22].set(-1)
    return a, b, jnp.stack((a, b), axis=1)


def test_full_image_choice_accepts_both_and_rejects_average_and_hybrids():
    a, b, family = candidates()
    t = pyramid(jnp.full_like(a, 0.5), 2)

    def costs(pred):
        return family_costs(pred, family, t)

    ca, cb = costs(pyramid(a, 2)), costs(pyramid(b, 2))
    avg = costs(pyramid((a + b) / 2, 2))
    assert float(ca.min()) == float(cb.min()) == 0
    assert float(avg.min()) > 0 and float(avg.mean()) < float(ca.mean())
    hybrid = a.at[:, 16:].set(b[:, 16:])
    assert float(costs(pyramid(hybrid, 2)).min()) > 0
    assert float(costs((a, pyramid(b, 2)[1])).min()) > 0
    np.testing.assert_allclose(ca, family_costs(pyramid(a, 2), family[:, ::-1], t)[:, ::-1])
    fine = family_costs((a,), family, (t[0],))
    coarse = family_costs(
        (pyramid(b, 2)[1],),
        pyramid(family.reshape(2, 32, 32, 1), 2)[1].reshape(1, 2, 16, 16, 1),
        (t[1],),
    )
    np.testing.assert_allclose(costs((a, pyramid(b, 2)[1])), (fine + 0.5 * coarse) / 1.5)


def test_protected_pixels_have_zero_gradient_and_designated_teacher_is_unused():
    a, b, family = candidates()
    t = jnp.full_like(a, 0.5).at[:, :, :16].set(0)
    grad = jax.grad(lambda x: family_costs((x,), family, (t,)).min())((a + b) / 2)
    assert np.all(np.asarray(grad)[:, :, :16] == 0)
    config, p, _ = fixture()
    model = fresh_model(config)
    for method in ("mean", "minimum"):
        fn = family_gradient(method)
        parts, grads = fn(model, p, family)
        other = p._replace(targets=tuple(-v for v in p.targets))
        jax.tree.map(np.testing.assert_array_equal, (parts, grads), fn(model, other, family))


def test_actual_updates_and_checkpoint_restore(tmp_path):
    config, p, _ = fixture()
    _, _, family = candidates()
    arms = {}
    for method in ("mean", "minimum"):
        arms[method] = make_arm(config, fresh_model(config))
        arms[method]["gradient"] = family_gradient(method)
        result = update(arms[method], p, family)
        assert result["gradient_norm"] > 0
    with checkpoints.manager(tmp_path / "saved") as saved:
        lineart_state.save(saved, arms, 1)
        restored = {}
        for method in arms:
            restored[method] = make_arm(config, fresh_model(config))
            restored[method]["gradient"] = family_gradient(method)
        assert lineart_state.restore(saved, restored) == 1
    for method in arms:
        assert update(arms[method], p, family) == update(restored[method], p, family)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 2),
        lineart_state.payload(restored, 2),
    )
