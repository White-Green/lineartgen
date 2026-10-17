import jax
import jax.numpy as jnp
import numpy as np
from one_sided_opening_objective import one_sided_consistency
from opening_prior_objective import opening_consistency


def fixture():
    # Non-saturated alpha keeps the renderer's derivative visible. Broad correct
    # stroke plus a narrow unwanted rail on a varying low-alpha background.
    rng = np.random.default_rng(2026101705)
    alpha = rng.uniform(0.01, 0.06, (1, 64, 64, 1)).astype(np.float32)
    alpha[:, 8:56, 28:36] = 0.95
    alpha[:, 8:56, 13:16] = 0.9
    target_alpha = np.zeros_like(alpha)
    target_alpha[:, 8:56, 28:36] = 1
    return (
        jnp.asarray((0.5 - alpha) / 4),
        jnp.asarray(((0.5 - target_alpha) / 4)[:, None]),
        jnp.ones((1, 16, 16, 1)),
    )


def test_value_preserved_and_gradient_cannot_add_ink_in_field_space():
    p, target, strength = fixture()
    before, old_gradient = jax.value_and_grad(opening_consistency)(p, target, strength)
    after, new_gradient = jax.value_and_grad(one_sided_consistency)(p, target, strength)
    np.testing.assert_array_equal(after, before)
    assert float(after) > 0
    assert np.isfinite(np.asarray(new_gradient)).all()
    assert float(new_gradient.max()) <= 0
    assert float(new_gradient.min()) < 0
    # The ordinary opening gradient also has a component that decreases field
    # values (adds ink) under gradient descent, unlike the detached version.
    assert float(old_gradient.max()) > 0
    assert float(old_gradient.min()) < 0


def test_permitted_targets_and_protected_region_are_exempt():
    p, target, strength = fixture()
    for candidate in (target[:, 0], p):
        candidates = jnp.concatenate([target, candidate[:, None]], axis=1)
        value, gradient = jax.value_and_grad(one_sided_consistency)(candidate, candidates, strength)
        assert float(value) == 0
        np.testing.assert_array_equal(gradient, jnp.zeros_like(candidate))
    for mask in (jnp.zeros_like(strength), strength.at[:, :, :8].set(0)):
        _, gradient = jax.value_and_grad(one_sided_consistency)(p, target, mask)
        protected = np.repeat(np.repeat(np.asarray(mask == 0), 4, 1), 4, 2)
        np.testing.assert_array_equal(np.asarray(gradient)[protected], 0)
        np.testing.assert_array_equal(
            one_sided_consistency(p, target, mask), opening_consistency(p, target, mask)
        )


def test_d4_transforms_preserve_value_and_gradient():
    p, target, strength = fixture()
    evaluate = jax.jit(jax.value_and_grad(one_sided_consistency))
    value, gradient = evaluate(p, target, strength)
    for flip in (False, True):
        for turns in range(4):

            def transform(a, axes, flip=flip, turns=turns):
                return jnp.rot90(jnp.flip(a, axes[1]) if flip else a, turns, axes)

            transformed_value, transformed_gradient = evaluate(
                transform(p, (1, 2)), transform(target, (2, 3)), transform(strength, (1, 2))
            )
            np.testing.assert_allclose(transformed_value, value, atol=1e-7, rtol=1e-6)
            np.testing.assert_allclose(
                transformed_gradient, transform(gradient, (1, 2)), atol=1e-7, rtol=1e-6
            )
