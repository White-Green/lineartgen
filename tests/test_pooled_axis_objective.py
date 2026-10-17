import jax.numpy as jnp
import numpy as np
import pytest
from axis_family_objective import axis_parts as original_parts
from plausible_study import exact
from pooled_axis_objective import axis_parts
from test_axis_family import candidates


@pytest.mark.parametrize("mode", ["none", "ordinary", "one_sided"])
def test_partial_guard_and_permitted_axis(mode):
    h, v, strengths = candidates()
    eligible = jnp.array([False, False])
    both = jnp.array([True, True])
    accepted = axis_parts((v[:, 0],), (h,), (h,), (v,), strengths, eligible, both, prior_mode=mode)
    assert float(accepted["loss"]) < 1e-6
    partial = (strengths[0].at[0, :, :8].set(0),)
    actual = axis_parts((v[:, 0],), (h,), (h,), (v,), partial, eligible, both, prior_mode=mode)
    expected = axis_parts(
        (v[:, 0],), (h,), (h,), (v,), partial, eligible, jnp.array([False, True]), prior_mode=mode
    )
    exact(actual, expected)
    assert float(actual["axis_eligible_fraction"]) == 0.5


def test_ordinary_matches_existing_and_scalar_subtraction():
    h, v, strengths = candidates()
    eligible, allowed = jnp.array([False, False]), jnp.array([True, False])
    pred = (v[:, 0] * 0.9,)
    original = original_parts(pred, (h,), (h,), (v,), strengths, eligible, allowed)
    ordinary = axis_parts(
        pred, (h,), (h,), (v,), strengths, eligible, allowed, prior_mode="ordinary"
    )
    exact(original, ordinary)
    one_sided = axis_parts(
        pred, (h,), (h,), (v,), strengths, eligible, allowed, prior_mode="one_sided"
    )
    exact(one_sided, ordinary)
    no_prior = axis_parts(pred, (h,), (h,), (v,), strengths, eligible, allowed, prior_mode="none")
    np.testing.assert_allclose(
        no_prior["loss"], original["loss"] - 0.1 * original["opening_consistency"], atol=1e-7
    )


def test_choice_is_joint_across_stages_and_cannot_mix_axes_per_pixel():
    h, v, strengths = candidates()
    eligible, allowed = jnp.array([False, False]), jnp.array([True, True])

    def loss(fields):
        return axis_parts(
            fields, (h, h), (h, h), (v, v), strengths * 2, eligible, allowed, prior_mode="none"
        )["loss"]

    assert float(loss((h[:, 0], h[:, 0]))) < 1e-6
    assert float(loss((v[:, 0], v[:, 0]))) < 1e-6
    assert float(loss((h[:, 0], v[:, 0]))) > 0.01
    cross = jnp.minimum(h[:, 0], v[:, 0])
    assert float(loss((cross, cross))) > 0.01
