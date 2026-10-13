import jax
import jax.numpy as jnp
import numpy as np
from cue_capacity import make_arm
from local_recursive_fit import fresh_model
from path_profile_data import path_probes, training_annotation
from path_profile_objective import path_costs, profile_gradient
from stable_assignment_study import update
from test_family_objective import candidates
from test_plausible_lineart import fixture

from lineartgen.diffusion import pyramid


def annotation():
    probes = np.zeros((1, 68, 6), np.float32)
    probes[:, :, 4] = 10
    probes[0, :17, 0] = 16
    probes[0, :17, 1] = np.linspace(5, 26, 17)
    probes[0, :17, 2] = 1
    valid = np.zeros((1, 68), np.float32)
    valid[0, :17] = 1
    return dict(probes=jnp.asarray(probes), valid=jnp.asarray(valid))


def test_arc_profiles_cover_both_corner_legs_and_empty_case():
    points = np.r_[
        np.c_[np.linspace(0, 64, 129), np.zeros(129)],
        np.c_[np.full(128, 64), np.linspace(0.5, 64, 128)],
    ]
    m = dict(kinds=["corner"], path_geometry=[dict(points=points.tolist(), widths=[2] * 257)])
    p, c = path_probes(m)
    assert len(p) == 17 and (p[:, 0] > 95).any() and (p[:, 1] > 89).any()
    assert c[-1, 0] < 40 and c[-1, 3] > 89
    assert training_annotation(dict(kinds=["blank"]))["valid"].sum() == 0


def test_profile_family_coherence_and_zero_protected_gradient():
    a, b, v = candidates()
    ann = annotation()
    t = pyramid(jnp.full_like(a, 0.5), 2)
    ca = path_costs(pyramid(a, 2), v, t, ann)
    cb = path_costs(pyramid(b, 2), v, t, ann)
    assert float(ca.min()) == float(cb.min()) == 0
    assert float(path_costs(pyramid((a + b) / 2, 2), v, t, ann).min()) > 0
    assert float(path_costs((a, pyramid(b, 2)[1]), v, t, ann).min()) > 0
    np.testing.assert_allclose(ca, path_costs(pyramid(a, 2), v[:, ::-1], t, ann)[:, ::-1])
    protected = t[0].at[:, :, :16].set(0)
    grad = jax.grad(lambda x: path_costs((x,), v, (protected,), ann).min())((a + b) / 2)
    assert np.all(np.asarray(grad)[:, :, :16] == 0)
    empty = ann | dict(valid=jnp.zeros_like(ann["valid"]))
    np.testing.assert_array_equal(path_costs(pyramid(a, 2), v, t, empty), jnp.zeros((1, 2)))


def test_thin_and_wide_missing_paths_have_equal_cost():
    clean = jnp.ones((1, 64, 64, 1)).at[:, 4:60, 16:17].set(-1).at[:, 4:60, 44:52].set(-1)
    ann = annotation()
    p = ann["probes"].at[0, :34, 1].set(32).at[0, 17:34, 0].set(48).at[0, 17:34, 2].set(1)
    ann = dict(probes=p, valid=ann["valid"].at[0, 17:34].set(1))
    thin = clean.at[:, :, 16:17].set(1)
    wide = clean.at[:, :, 44:52].set(1)
    t = (jnp.full_like(clean, 0.5),)
    a = path_costs((thin,), clean[:, None], t, ann)
    b = path_costs((wide,), clean[:, None], t, ann)
    np.testing.assert_allclose(a, b, rtol=0.04)
    assert float(a[0, 0]) > 0.4


def test_profile_actual_update_teacher_unused():
    config, p, _ = fixture()
    _, _, v = candidates()
    ann = annotation()
    for weight in (0.0, 0.25):
        arm = make_arm(config, fresh_model(config))
        arm["gradient"] = profile_gradient(weight)
        other = p._replace(targets=tuple(-x for x in p.targets))
        jax.tree.map(
            np.testing.assert_array_equal,
            arm["gradient"](arm["model"], p, v, ann),
            arm["gradient"](arm["model"], other, v, ann),
        )
        assert update(arm, p, v, ann)["gradient_norm"] > 0
