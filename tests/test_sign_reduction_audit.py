import jax
import jax.numpy as jnp
import numpy as np
from consistent_axis_objective import field_parts as original_parts
from consistent_axis_objective import sign_mean
from local_sign_objective import sign_mean as local_sign
from sign_audit_objective import field_parts
from sign_reduction_audit import region_gradient


def test_exposed_costs_preserve_original_value_choice_and_gradient():
    rng = np.random.default_rng(41)
    fields = tuple(
        jnp.asarray(rng.normal(size=(2, 32 // 2**i, 32 // 2**i, 1)), dtype=jnp.float32)
        for i in range(3)
    )
    targets = tuple(
        jnp.asarray(rng.normal(size=(2, 9, 32 // 2**i, 32 // 2**i, 1)) * 2, dtype=jnp.float32)
        for i in range(3)
    )
    strengths = tuple(jnp.ones((2, 8 // 2**i, 8 // 2**i, 1)) for i in range(3))
    a, b = original_parts(fields, targets, strengths), field_parts(fields, targets, strengths)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    np.testing.assert_array_equal(b["choice"][:, 0], np.argmin(b["candidate_costs"], axis=1))
    g1 = jax.grad(lambda ff: original_parts(ff, targets, strengths)["loss"])(fields)
    g2 = jax.grad(lambda ff: field_parts(ff, targets, strengths)["loss"])(fields)
    jax.tree.map(np.testing.assert_array_equal, g1, g2)


def test_analytic_gradient_uses_shared_candidate_denominator():
    v = np.linspace(0.01, 0.3, 64).reshape(8, 8)
    masks = np.ones((2, 8, 8), bool)
    masks[0, :3] = False
    denominator = max(masks.sum((1, 2)).max(), 1)

    def fn(x):
        vv = jnp.broadcast_to(x[None, None, ..., None], (1, 2, 8, 8, 1))
        return sign_mean(
            vv, jnp.asarray(masks[None, ..., None]), jnp.asarray([[denominator]]), power=4
        )[0, 0]

    analytic, _, _ = region_gradient(v, masks, 0)
    np.testing.assert_allclose(
        analytic, jax.grad(fn)(jnp.asarray(v, dtype=jnp.float32)), rtol=2e-6, atol=1e-8
    )


def test_sparse_small_violation_reveals_epsilon_damping():
    v = np.zeros((16, 16))
    v[0, 0] = 0.01
    masks = np.ones((1, 16, 16), bool)
    a, m, n = region_gradient(v, masks, 0)
    b, _, _ = region_gradient(v, masks, 0, epsilon=0.0001)
    ratio = ((m[0, 0] + 0.01**4) / (m[0, 0] + 0.0001**4)) ** 0.75
    np.testing.assert_allclose(b[0, 0] / a[0, 0], ratio, rtol=1e-12)
    assert ratio > 50 and n[0, 0] == 256


def test_local_gradient_decouples_distant_peak_and_matches_existing_reduction():
    v = np.zeros((16, 16))
    v[0, 0] = 0.15
    masks = np.ones((2, 16, 16), bool)

    def fn(x):
        vv = jnp.broadcast_to(x[None, None, ..., None], (1, 2, 16, 16, 1))
        return local_sign(vv, jnp.asarray(masks[None, ..., None]), None, power=4)[0, 0]

    a, _, _ = region_gradient(v, masks, 0, tiles=4)
    np.testing.assert_allclose(
        a, jax.grad(fn)(jnp.asarray(v, dtype=jnp.float32)), rtol=2e-6, atol=1e-8
    )
    ga, _, _ = region_gradient(v, masks, 0)
    v[15, 15] = 2
    b, _, _ = region_gradient(v, masks, 0, tiles=4)
    gb, _, _ = region_gradient(v, masks, 0)
    assert a[0, 0] == b[0, 0] and gb[0, 0] < ga[0, 0] / 100


def test_empty_region_has_zero_gradient():
    g, m, n = region_gradient(np.ones((8, 8)), np.zeros((2, 8, 8), bool), 0, tiles=4)
    assert not g.any() and not m.any() and not n.any()
