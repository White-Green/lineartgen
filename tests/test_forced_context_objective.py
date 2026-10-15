import jax
import jax.numpy as jnp
import numpy as np
from forced_context_objective import family_parts
from plausible_study import exact
from quartic_context_objective import family_parts as legacy_parts
from quartic_field_objective import field_parts


def fixture():
    y = (jnp.arange(64) + 0.5) / 4
    merged = jnp.broadcast_to((abs(y - 8) - 0.8)[None, :, None, None], (1, 64, 64, 1))
    original = jnp.minimum(abs(merged + 0.8 - 2) - 0.8, abs(y[None, :, None, None] - 6) - 0.8)
    return original, merged, jnp.ones((1, 16, 16, 1)) * 0.5


def test_optional_legacy_selection_reproduces_loss_and_gradient():
    original, merged, t = fixture()
    a, b = original[:, None], merged[:, None]
    eligible = jnp.ones(1, dtype=bool)
    pred = original + 0.15
    old = jax.value_and_grad(lambda x: legacy_parts((x,), (a,), (b,), (t,), eligible)["loss"])(pred)
    new = jax.value_and_grad(
        lambda x: family_parts((x,), (a,), (b,), (t,), eligible, force_merge=False)["loss"]
    )(pred)
    exact(old, new)


def test_forced_choice_matches_merged_even_when_original_is_closer():
    original, merged, t = fixture()
    a, b = original[:, None], merged[:, None]
    eligible = jnp.ones(1, dtype=bool)
    old = legacy_parts((original,), (a,), (b,), (t,), eligible)
    assert float(old["merged_fraction"]) == 0
    actual = jax.value_and_grad(lambda x: family_parts((x,), (a,), (b,), (t,), eligible)["loss"])(
        original
    )
    expected = jax.value_and_grad(lambda x: field_parts((x,), (b,), (t,))["loss"])(original)
    exact(actual, expected)
    assert float(actual[0]) > 0 and np.isfinite(np.asarray(actual[1])).all()
    parts = family_parts((merged,), (a,), (b,), (t,), eligible)
    assert float(parts["merged_fraction"]) == 1
    np.testing.assert_allclose(parts["loss"], 0, atol=1e-7)


def test_unsupported_families_and_protected_pixels_remain_unchanged():
    original, merged, t = fixture()
    a, b = original[:, None], merged[:, None]
    no = jnp.zeros(1, dtype=bool)
    old = jax.value_and_grad(lambda x: legacy_parts((x,), (a,), (b,), (t,), no)["loss"])(
        original + 0.2
    )
    new = jax.value_and_grad(lambda x: family_parts((x,), (a,), (b,), (t,), no)["loss"])(
        original + 0.2
    )
    exact(old, new)
    t = t.at[:, :, :8].set(0)
    pred = merged.at[:, :, :32].set(-3)
    loss, grad = jax.value_and_grad(lambda x: family_parts((x,), (a,), (b,), (t,), ~no)["loss"])(
        pred
    )
    np.testing.assert_allclose(loss, 0, atol=1e-7)
    np.testing.assert_allclose(grad, 0, atol=1e-7)
