import jax
import jax.numpy as jnp
import numpy as np
from coherent_broad_objective import axis_parts
from support_margin_objective import support_margin, support_parts
from test_consistent_broad import toy


def fixture():
    y, x = jnp.mgrid[:32, :32]
    a = jnp.abs(x - 12) / 4 - 1
    b = jnp.abs(x - 16) / 4 - 1
    return jnp.stack((a, b))[None, ..., None], jnp.ones((1, 8, 8, 1)) * 0.5


def test_every_permitted_teacher_and_protected_output_have_zero_extra_cost():
    t, s = fixture()
    for i in range(2):
        f = t[:, i]
        assert float(support_margin(f, t, s)) == 0
        np.testing.assert_array_equal(jax.grad(support_margin)(f, t, s), 0)
    f = jnp.full_like(t[:, 0], -8)
    assert float(support_margin(f, t, s * 0)) == 0
    np.testing.assert_array_equal(jax.grad(support_margin)(f, t, s * 0), 0)


def test_gray_and_saturated_black_get_equal_whiteward_gradients_only_outside_support():
    t, s = fixture()
    f = jnp.ones_like(t[:, 0]).at[0, 4, 1, 0].set(0.1).at[0, 5, 1, 0].set(-8)
    f = f.at[0, 6, 12, 0].set(-8)
    g = jax.grad(support_margin)(f, t, s)
    assert g[0, 4, 1, 0] < 0 and g[0, 4, 1, 0] == g[0, 5, 1, 0]
    assert g[0, 6, 12, 0] == 0 and g[0, 7, 1, 0] == 0
    partial = s.at[:, :2].set(0)
    np.testing.assert_array_equal(jax.grad(support_margin)(f, t, partial)[:, :8], 0)


def test_blank_fill_duplicate_and_permuted_candidates_are_finite_and_consistent():
    t, s = fixture()
    f = jnp.zeros_like(t[:, 0])
    a = support_margin(f, t, s)
    assert a > 0
    np.testing.assert_array_equal(a, support_margin(f, t[:, ::-1], s))
    np.testing.assert_array_equal(a, support_margin(f, jnp.repeat(t, 2, axis=1), s))
    for v in (-8.0, 8.0):
        q = jnp.full_like(t, v)
        loss, grad = jax.value_and_grad(support_margin)(f, q, s)
        assert np.isfinite(loss) and np.isfinite(grad).all()
        if v < 0:
            assert loss == 0


def test_zero_weight_preserves_all_old_parts_and_field_gradients():
    f, fam, strength, normal = toy()
    eligible = jnp.array([False, True])
    for allowed in (jnp.zeros(2, bool), jnp.array([True, False])):
        args = (*fam, strength, eligible, allowed, normal)
        old = axis_parts(f, *args, policy="retained", opening_weight=0.1)
        new = support_parts(f, *args, weight=0.0)
        for k in old:
            np.testing.assert_array_equal(old[k], new[k])
        ga = jax.grad(
            lambda ff, args=args: axis_parts(ff, *args, policy="retained", opening_weight=0.1)[
                "loss"
            ]
        )(f)
        gb = jax.grad(lambda ff, args=args: support_parts(ff, *args, weight=0.0)["loss"])(f)
        jax.tree.map(np.testing.assert_array_equal, ga, gb)


def test_added_support_term_does_not_change_assignment_or_other_parts():
    f, fam, strength, normal = toy()
    args = (*fam, strength, jnp.array([False, True]), jnp.array([True, False]), normal)
    a = support_parts(f, *args, weight=0.0)
    b = support_parts(f, *args, weight=0.1)
    for k in a:
        if k != "loss":
            np.testing.assert_array_equal(a[k], b[k])
    np.testing.assert_allclose(b["loss"] - a["loss"], 0.1 * a["support_margin"], atol=1e-7)
