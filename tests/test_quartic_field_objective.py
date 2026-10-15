import jax
import jax.numpy as jnp
import numpy as np
from plausible_study import exact
from quartic_field_objective import field_parts
from regional_field_objective import field_parts as old_parts


def fixture():
    yy = (jnp.arange(64) + 0.5) / 4
    target = jnp.broadcast_to((abs(yy - 8) - 0.8)[None, :, None, None], (1, 64, 64, 1))
    candidates = jnp.stack((target, target + 0.5), axis=1)
    t = jnp.ones((1, 16, 16, 1)) * 0.5
    return target, candidates, t


def test_rms_reproduces_existing_loss_and_gradient():
    pred, targets, t = fixture()
    pred = pred + jax.random.normal(jax.random.PRNGKey(3), pred.shape) * 0.2
    old = jax.value_and_grad(lambda p: old_parts((p,), (targets,), (t,))["loss"])(pred)
    new = jax.value_and_grad(lambda p: field_parts((p,), (targets,), (t,), sign_power=2)["loss"])(
        pred
    )
    exact(old, new)


def test_correct_family_and_protected_region_have_zero_penalty():
    pred, targets, t = fixture()
    for p in (pred, pred + 0.5):
        loss, grad = jax.value_and_grad(lambda x: field_parts((x,), (targets,), (t,))["loss"])(p)
        np.testing.assert_allclose(loss, 0, atol=1e-7)
        np.testing.assert_allclose(grad, 0, atol=1e-7)
    t = t.at[:, :, :8].set(0)
    pred = pred.at[:, :, :32].set(-3)
    loss, grad = jax.value_and_grad(lambda x: field_parts((x,), (targets,), (t,))["loss"])(pred)
    np.testing.assert_allclose(loss, 0, atol=1e-7)
    np.testing.assert_allclose(grad, 0, atol=1e-7)


def test_sparse_gap_and_ghost_receive_stronger_corrective_sign_gradient():
    clean, targets, t = fixture()
    # One wrong-white gap within a line; one wrong-black island far outside it.
    for region, wrong, direction in [
        ((slice(None), slice(30, 34), slice(30, 34), slice(None)), 1.0, 1),
        ((slice(None), slice(1, 3), slice(20, 24), slice(None)), -1.0, -1),
    ]:
        pred = clean.at[region].set(wrong)
        gradients = {}
        for power in (2, 4):
            loss, grad = jax.value_and_grad(
                lambda x, power=power: field_parts((x,), (targets,), (t,), sign_power=power)[
                    "sign_margin"
                ]
            )(pred)
            assert np.isfinite(float(loss)) and np.isfinite(np.asarray(grad)).all()
            gradients[power] = float(grad[region].sum()) * direction
            assert gradients[power] > 0
        assert gradients[4] > gradients[2] * 1.5
