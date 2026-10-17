import jax
import jax.numpy as jnp
import numpy as np
from consistent_broad_objective import allowed_mask, axis_parts
from pooled_axis_objective import axis_parts as old_parts


def toy():
    rng = np.random.default_rng(193)
    fields = tuple(
        jnp.asarray(rng.normal(0, 0.4, (2, s, s, 1)).astype(np.float32)) for s in (16, 8)
    )
    families = [
        tuple(jnp.asarray(rng.normal(0, 0.8, (2, 9, s, s, 1)).astype(np.float32)) for s in (16, 8))
        for _ in range(3)
    ]
    t = tuple(jnp.ones((2, s // 4, s // 4, 1)) * 0.5 for s in (16, 8))
    normal = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    return fields, families, t, normal


def test_whole_and_eligibility_guard():
    _, _, t, n = toy()
    for allowed in (jnp.array([False, False]), jnp.array([True, False]), jnp.array([True, True])):
        mask = np.asarray(allowed_mask(n, allowed, t, "consistent"))
        assert (mask.sum(1) == 9).all()
        assert mask[0, 9] == bool(allowed[0]) and mask[1, 0]
    t = (t[0].at[0, 0, 0, 0].set(0), t[1])
    assert np.asarray(allowed_mask(n, jnp.ones(2, bool), t, "consistent"))[0, :9].all()


def test_winner_matches_original_cost_and_gradients_for_both_branches():
    f, fam, t, n = toy()
    eligible = jnp.array([False, True])
    for allowed in (jnp.array([False, False]), jnp.array([True, False])):
        a = old_parts(f, *fam, t, eligible, allowed, prior_mode="none")
        b = axis_parts(f, *fam, t, eligible, allowed, n, policy="winner")
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])
        ga = jax.grad(
            lambda x, permitted=allowed: old_parts(
                x, *fam, t, eligible, permitted, prior_mode="none"
            )["loss"]
        )(f)
        gb = jax.grad(
            lambda x, permitted=allowed: axis_parts(
                x, *fam, t, eligible, permitted, n, policy="winner"
            )["loss"]
        )(f)
        for aa, bb in zip(ga, gb, strict=True):
            np.testing.assert_array_equal(aa, bb)


def test_consistent_without_allowed_axes_keeps_original_full_gradient():
    f, fam, t, n = toy()
    eligible = jnp.array([False, True])
    allowed = jnp.zeros(2, bool)

    def old(x):
        return old_parts(x, *fam, t, eligible, allowed, prior_mode="none")["loss"]

    def new(x):
        return axis_parts(x, *fam, t, eligible, allowed, n, policy="consistent")["loss"]

    np.testing.assert_array_equal(old(f), new(f))
    for aa, bb in zip(jax.grad(old)(f), jax.grad(new)(f), strict=True):
        np.testing.assert_array_equal(aa, bb)
