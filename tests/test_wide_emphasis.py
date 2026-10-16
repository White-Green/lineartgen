"""Verify weighting against independent singleton objectives, including gradients."""

import jax
import jax.numpy as jnp
import numpy as np
from forced_context_objective import family_parts
from opening_prior_objective import opening_consistency
from wide_emphasis_objective import emphasized_parts


def fixture():
    rng = np.random.default_rng(71)
    f = tuple(jnp.asarray(rng.normal(size=(2, n, n, 1)), dtype=jnp.float32) for n in (8, 4))
    a = tuple(jnp.asarray(rng.normal(size=(2, 9, n, n, 1)), dtype=jnp.float32) for n in (8, 4))
    b = tuple(-x for x in a)
    s = (jnp.full((2, 2, 2, 1), 0.5), jnp.full((2, 1, 1, 1), 0.5))
    return f, a, b, s, jnp.asarray([False, True])


def independent(fields, a, b, s, e, w):
    total = 0.0
    for i in range(2):
        ff = tuple(x[i : i + 1] for x in fields)
        aa = tuple(x[i : i + 1] for x in a)
        bb = tuple(x[i : i + 1] for x in b)
        ss = tuple(x[i : i + 1] for x in s)
        parts = family_parts(ff, aa, bb, ss, e[i : i + 1])
        teacher = bb[0] if bool(e[i]) else aa[0]
        total += w[i] * (parts["loss"] + 0.1 * opening_consistency(ff[0], teacher, ss[0])) / 2
    return total


def test_weighted_loss_and_gradient_match_singleton_objectives():
    f, a, b, s, e = fixture()
    w = jnp.asarray([5.0, 1.0]) / 1.204

    def actual(x):
        return emphasized_parts(x, a, b, s, e, w)["loss"]

    def expected(x):
        return independent(x, a, b, s, e, w)

    np.testing.assert_allclose(actual(f), expected(f), atol=2e-6, rtol=2e-6)
    for x, y in zip(jax.grad(actual)(f), jax.grad(expected)(f), strict=True):
        np.testing.assert_allclose(x, y, atol=2e-6, rtol=2e-5)


def test_neutral_weights_match_existing_batch_objective():
    f, a, b, s, e = fixture()
    w = jnp.ones((2,))
    selected = jnp.where(e[:, None, None, None, None], b[0], a[0])
    expected = family_parts(f, a, b, s, e)["loss"] + 0.1 * opening_consistency(f[0], selected, s[0])
    np.testing.assert_allclose(
        emphasized_parts(f, a, b, s, e, w)["loss"], expected, atol=2e-6, rtol=2e-6
    )

    def original_loss(x):
        return family_parts(x, a, b, s, e)["loss"] + 0.1 * opening_consistency(x[0], selected, s[0])

    def neutral_loss(x):
        return emphasized_parts(x, a, b, s, e, w)["loss"]

    for x, y in zip(jax.grad(original_loss)(f), jax.grad(neutral_loss)(f), strict=True):
        np.testing.assert_allclose(x, y, atol=2e-6, rtol=2e-5)


def test_inactive_examples_have_no_weighted_loss_or_gradient():
    f, a, b, s, e = fixture()
    s = tuple(jnp.zeros_like(x) for x in s)
    w = jnp.asarray([5.0, 1.0]) / 1.204

    def loss(x):
        return emphasized_parts(x, a, b, s, e, w)["loss"]

    np.testing.assert_allclose(loss(f), 0, atol=1e-7)
    for x in jax.grad(loss)(f):
        np.testing.assert_array_equal(x, jnp.zeros_like(x))
