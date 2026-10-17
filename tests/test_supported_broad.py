import jax
import jax.numpy as jnp
import numpy as np
from consistent_broad_objective import axis_parts as old_parts
from supported_broad_objective import axis_parts, effective_permission
from test_consistent_broad import toy
from test_local_axis_support import case


def test_partial_permission_requires_metadata_and_complete_support():
    fields, t = case()
    a, b = jnp.asarray(fields[:, :9]), jnp.asarray(fields[:, 9:])
    assert bool(effective_permission(jnp.array([True]), a, b, jnp.asarray(t))[0])
    assert not bool(effective_permission(jnp.array([False]), a, b, jnp.asarray(t))[0])
    b = b.at[:, 8, 11, 29].set(3.9)
    assert not bool(effective_permission(jnp.array([True]), a, b, jnp.asarray(t))[0])


def test_unmodified_whole_and_forbidden_batches_keep_costs_and_gradients():
    f, fam, t, n = toy()
    eligible = jnp.array([False, True])
    for allowed, strength in (
        (jnp.array([True, False]), t),
        (jnp.zeros(2, bool), (t[0].at[:, 0, 0].set(0), t[1])),
    ):
        args = (*fam, strength, eligible, allowed, n)
        a = old_parts(f, *args, policy="consistent")
        b = axis_parts(f, *args, policy="consistent")
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])
        old_grad = jax.grad(
            lambda ff, args=args: old_parts(ff, *args, policy="consistent")["loss"]
        )(f)
        new_grad = jax.grad(
            lambda ff, args=args: axis_parts(ff, *args, policy="consistent")["loss"]
        )(f)
        for aa, bb in zip(old_grad, new_grad, strict=True):
            np.testing.assert_array_equal(aa, bb)


def test_partial_nonempty_requirement_does_not_allow_blank_alternate():
    fields, t = case()
    a = jnp.asarray(fields[:, :9])
    empty = jnp.full_like(a, 10)
    assert not bool(effective_permission(jnp.array([True]), a, empty, jnp.asarray(t))[0])


def test_complete_partial_support_reaches_final_axis_selection():
    fields, t = case()
    aa, bb = (jnp.asarray(fields[:, :9]),), (jnp.asarray(fields[:, 9:]),)
    pred = (aa[0][:, 0],)
    args = (
        aa,
        aa,
        bb,
        (jnp.asarray(t),),
        jnp.array([False]),
        jnp.array([True]),
        jnp.array([[1.0, 0.0]]),
    )
    assert float(old_parts(pred, *args, policy="consistent")["selected_alternate_bits"]) == 0
    assert float(axis_parts(pred, *args, policy="consistent")["selected_alternate_bits"]) == 1
