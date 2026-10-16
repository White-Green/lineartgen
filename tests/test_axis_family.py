import copy

import jax
import jax.numpy as jnp
import numpy as np
from axis_family_data import axis_metadata, axis_spec
from axis_family_objective import axis_parts
from forced_context_objective import family_parts
from opening_prior_objective import opening_consistency
from plausible_study import exact


def record():
    points = np.stack((np.linspace(20, 80, 129), np.full(129, 50.0)), -1)
    return dict(
        kind="wide",
        metadata=dict(
            path_geometry=[dict(kind="straight", points=points.tolist(), widths=[32.0] * 129)]
        ),
    )


def test_only_short_isolated_wide_gets_perpendicular_interpretation():
    r = record()
    original = copy.deepcopy(r)
    assert not axis_spec(r, 64.0, whole=False)["allowed"]
    assert not axis_spec(r, 44.9, whole=True)["allowed"]
    s = axis_spec(r, 45.0, whole=True)
    assert s["allowed"] and s["ratio"] == 0.75
    assert not axis_spec(r, 64.0, overlay=True, whole=True)["allowed"]
    thin = copy.deepcopy(r)
    thin["kind"] = "straight"
    assert not axis_spec(thin, 64.0, whole=True)["allowed"]
    points = np.asarray(s["points"])
    np.testing.assert_allclose(points[[0, -1]], [[50.0, 27.5], [50.0, 72.5]])
    np.testing.assert_allclose(s["direction"], [-1.0, 0.0])
    options = dict(amplitude=0.0, period=64.0, phase_x=0.0, phase_y=0.0)
    # Zero displacement is the identity regardless of phase fields.
    m = axis_metadata(r, s, options)
    np.testing.assert_allclose(m["path_geometry"][0]["points"], points)
    assert m["path_geometry"][0]["widths"] == [2.0] * 129
    assert r == original


def candidates():
    q = (jnp.arange(64) + 0.5) / 4
    yy, xx = jnp.meshgrid(q, q, indexing="ij")
    horizontal = jnp.sqrt((xx - jnp.clip(xx, 4, 12)) ** 2 + (yy - 8) ** 2 + 1e-12) - 1
    vertical = jnp.rot90(horizontal)
    h = jnp.tile(horizontal[None, None, ..., None], (2, 9, 1, 1, 1))
    v = jnp.tile(vertical[None, None, ..., None], (2, 9, 1, 1, 1))
    return h, v, (jnp.full((2, 16, 16, 1), 0.5),)


def test_disabled_exact_and_whole_alternative_has_finite_gradient():
    h, v, strengths = candidates()
    eligible = jnp.array([False, False])
    disabled = jnp.array([False, False])
    pred = v[:, 0]
    baseline = family_parts((pred,), (h,), (h,), strengths, eligible)
    prior = opening_consistency(pred, h, strengths[0])
    baseline |= dict(opening_consistency=prior, loss=baseline["loss"] + 0.1 * prior)
    actual = axis_parts((pred,), (h,), (h,), (v,), strengths, eligible, disabled)
    actual.pop("axis_eligible_fraction")
    exact(actual, baseline)
    both = jnp.array([True, True])
    accepted = axis_parts((pred,), (h,), (h,), (v,), strengths, eligible, both)
    assert float(accepted["loss"]) < 1e-6
    assert float(baseline["loss"]) > 0.01
    one = jnp.array([True, False])
    partially = axis_parts((pred,), (h,), (h,), (v,), strengths, eligible, one)
    assert 0 < float(partially["loss"]) < float(baseline["loss"])
    for allowed in (disabled, both, one):
        grad = jax.grad(
            lambda f, a=allowed: axis_parts((f,), (h,), (h,), (v,), strengths, eligible, a)["loss"]
        )(pred)
        assert np.isfinite(np.asarray(grad)).all()
    protected = tuple(jnp.zeros_like(x) for x in strengths)
    assert float(axis_parts((pred,), (h,), (h,), (v,), protected, eligible, both)["loss"]) == 0


def test_partial_mask_defense_even_if_caller_allows_alternative():
    h, v, strengths = candidates()
    eligible = jnp.array([False, False])
    partial = (strengths[0].at[0, :, :8].set(0),)
    pred = v[:, 0]
    both = jnp.array([True, True])
    only_whole = jnp.array([False, True])
    a = axis_parts((pred,), (h,), (h,), (v,), partial, eligible, both)
    b = axis_parts((pred,), (h,), (h,), (v,), partial, eligible, only_whole)
    exact(a, b)
    assert float(a["axis_eligible_fraction"]) == 0.5
    all_partial = (partial[0].at[1, :, :8].set(0),)
    baseline = family_parts((pred,), (h,), (h,), all_partial, eligible)
    prior = opening_consistency(pred, h, all_partial[0])
    baseline |= dict(opening_consistency=prior, loss=baseline["loss"] + 0.1 * prior)
    actual = axis_parts((pred,), (h,), (h,), (v,), all_partial, eligible, both)
    assert float(actual.pop("axis_eligible_fraction")) == 0
    exact(actual, baseline)
