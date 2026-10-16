"""Check independent normalization and sparse-defect gradients, not model quality."""

import jax
import jax.numpy as jnp
import numpy as np
from local_sign_objective import sign_mean
from quartic_field_objective import sign_mean as global_mean


def test_other_terms_and_all_stage_weighting_are_unchanged():
    from local_sign_objective import field_parts as local_parts
    from quartic_field_objective import field_parts as global_parts

    fields = (jnp.full((1, 32, 32, 1), -0.2), jnp.full((1, 16, 16, 1), -0.4))
    targets = tuple(jnp.ones((1, 1, *x.shape[1:])) for x in fields)
    strengths = (jnp.ones((1, 8, 8, 1)), jnp.ones((1, 4, 4, 1)))
    a, b = local_parts(fields, targets, strengths), global_parts(fields, targets, strengths)
    for key in ("distance", "geometry_gradient"):
        np.testing.assert_array_equal(a[key], b[key])
    # Spatially constant violations must keep the original .5**level aggregate.
    np.testing.assert_allclose(a["sign_margin"], b["sign_margin"], atol=1e-7, rtol=1e-6)


def test_local_errors_do_not_compete_with_distant_large_error():
    mask = jnp.ones((1, 1, 16, 16, 1))
    values = jnp.zeros_like(mask).at[0, 0, 1, 1, 0].set(0.25).at[0, 0, 14, 14, 0].set(3)
    den = mask.sum((2, 3, 4))

    def grad(fn):
        return jax.grad(lambda v: fn(v, mask, den, power=4).sum())(values)

    local, global_ = grad(sign_mean), grad(global_mean)
    assert float(local[0, 0, 1, 1, 0]) > 10 * float(global_[0, 0, 1, 1, 0])
    assert float(local[0, 0, 14, 14, 0]) > 0
    assert np.isfinite(np.asarray(local)).all()


def test_shared_candidate_denominator_and_empty_tiles():
    mask = jnp.zeros((1, 2, 16, 16, 1)).at[0, 0, :4, :4, 0].set(1).at[0, 1, :2, :4, 0].set(1)
    v = jnp.ones_like(mask) * 0.5
    loss = sign_mean(v, mask, mask.sum((2, 3, 4)).max(1, keepdims=True), power=4)
    expected = (jnp.array([[1, 0.5]]) * 0.5**4 + 0.01**4) ** 0.25 - 0.01
    np.testing.assert_allclose(loss, expected, atol=1e-7, rtol=1e-6)
    empty = jnp.zeros_like(mask)
    np.testing.assert_allclose(sign_mean(v, empty, jnp.ones((1, 1)), power=4), 0, atol=1e-8)
    gradient = jax.grad(lambda x: sign_mean(x, empty, jnp.ones((1, 1)), power=4).sum())(v)
    np.testing.assert_array_equal(gradient, np.zeros_like(gradient))


def test_spatial_rotation_and_candidate_permutation():
    rng = np.random.default_rng(2026101601)
    v = jnp.asarray(rng.uniform(0, 2, (2, 3, 16, 16, 1)), jnp.float32)
    mask = jnp.asarray(rng.random(v.shape) > 0.3, jnp.float32)
    den = mask.sum((2, 3, 4)).max(1, keepdims=True)
    expected = sign_mean(v, mask, den, power=4)
    np.testing.assert_allclose(
        sign_mean(jnp.rot90(v, axes=(2, 3)), jnp.rot90(mask, axes=(2, 3)), den, power=4),
        expected,
        atol=1e-6,
    )
    order = jnp.array([2, 0, 1])
    np.testing.assert_allclose(
        sign_mean(v[:, order], mask[:, order], den, power=4), expected[:, order], atol=1e-6
    )
