import jax
import jax.numpy as jnp
import numpy as np
from axis_family_data import axis_payload
from coherent_broad_objective import axis_parts
from supported_broad_objective import axis_parts as supported_parts
from test_axis_family import candidates, record
from test_consistent_broad import toy


def test_unchanged_whole_and_disabled_costs_and_gradients():
    f, fam, t, n = toy()
    eligible = jnp.array([False, True])
    for allowed, strength in (
        (jnp.array([True, False]), t),
        (jnp.zeros(2, bool), (t[0].at[:, 0, 0].set(0), t[1])),
    ):
        args = (*fam, strength, eligible, allowed, n)
        old = supported_parts(f, *args, policy="consistent")
        new = axis_parts(f, *args, policy="retained")
        for k in old:
            np.testing.assert_array_equal(old[k], new[k])
        a = jax.grad(lambda v, args=args: supported_parts(v, *args, policy="consistent")["loss"])(f)
        b = jax.grad(lambda v, args=args: axis_parts(v, *args, policy="retained")["loss"])(f)
        for aa, bb in zip(a, b, strict=True):
            np.testing.assert_array_equal(aa, bb)


def test_active_tail_is_selected_instead_of_empty_original_and_is_recipe_independent():
    h, v, strengths = candidates()
    t = (strengths[0].at[:, :10].set(0),)
    eligible = jnp.zeros(2, bool)
    allowed = jnp.ones(2, bool)
    normal = jnp.array([[0.0, 1.0], [-1.0, 0.0]])
    original = jnp.stack((h[0], v[1]))
    alternate = jnp.stack((v[0], h[1]))
    pred = (v[:, 0],)
    args = ((original,), (original,), (alternate,), t, eligible, allowed, normal)
    parts = axis_parts(pred, *args, policy="retained")
    assert float(parts["selected_alternate_bits"]) == 1
    assert float(parts["loss"]) < 1e-6
    rejected = axis_parts((h[:, 0],), *args, policy="retained")
    assert float(rejected["loss"]) > float(parts["loss"]) + 0.001
    grad = jax.grad(lambda f: axis_parts(f, *args, policy="retained")["loss"])((h[:, 0],))
    assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(grad))


def test_legacy_partial_candidate_is_generated_only_when_explicitly_enabled():
    r = record()
    options = dict(amplitude=0.0, period=64.0, phase_x=0.0, phase_y=0.0)
    selection = dict(
        indices=[0],
        augmentation=[options],
        overlays=[dict(enabled=False)],
        transforms=[0],
        modes=[2],
    )
    base = jnp.full((1, 1056, 1056), 8, jnp.float16)
    original = (base, base, jnp.array([False]), jnp.ones((1, 1, 1, 1, 1)))
    normal = jnp.array([[0.0, 1.0]])
    args = (dict(records=[r]), selection, [dict(primary=dict(new=64.0))], original, normal)
    disabled, old_records = axis_payload(*args)
    enabled, new_records = axis_payload(*args, allow_partial=True)
    assert not old_records[0]["allowed"] and new_records[0]["allowed"]
    np.testing.assert_array_equal(disabled[0], base)
    assert enabled[0].shape == base.shape and enabled[0].dtype == base.dtype
    assert np.any(np.asarray(enabled[0]) < 0)
    np.testing.assert_array_equal(enabled[1], jnp.array([[-1.0, 0.0]]))
