import jax
import jax.numpy as jnp
import numpy as np
import pytest
from diagnose_learnability import update
from flax import nnx
from preconditioned_velocity import SIGNAL_STD, coefficients, objective, recursive_velocity
from preconditioned_velocity_fit import make_arms
from test_velocity_bridge_toy import fixture

from lineartgen.config import load_config, save_config
from lineartgen.training import checkpoints, lineart_state


@pytest.mark.parametrize("strength", [0.001, 0.1, 0.5, 1.0])
def test_coefficients_whiten_input_and_residual_at_both_endpoints(strength):
    tau = np.array([0, 0.0001, 0.1, 0.5, 0.99, 0.9999, 1], np.float32)
    cin, skip, cout = map(np.asarray, coefficients(jnp.full_like(tau, strength), 1 - tau))
    a2, b2 = SIGNAL_STD**2, (0.25 * strength) ** 2
    d = tau.astype(np.float64) ** 2 * a2 + (1 - tau.astype(np.float64)) ** 2 * b2
    cov = tau * a2 - (1 - tau) * b2
    # Evaluate residual variance as two squares, avoiding cancellation near tau=1.
    residual_variance = (1 - skip * tau) ** 2 * a2 + (-1 - skip * (1 - tau)) ** 2 * b2
    np.testing.assert_allclose(cin**2 * d, 1, atol=2e-4, rtol=2e-4)
    np.testing.assert_allclose(residual_variance / cout**2, 1, atol=2e-4, rtol=2e-4)
    np.testing.assert_allclose((cov - skip * d) * cin / cout, 0, atol=2e-4, rtol=0)
    assert np.isfinite(cin).all() and np.isfinite(skip).all() and np.all(cout > 0)


class ConstantNetwork(nnx.Module):
    def __init__(self):
        self.value = nnx.Param(jnp.array(0.2))

    def __call__(self, image, level, insert, source, progress, return_network_output=False):
        raw = jnp.full_like(image, self.value[...]) + insert.mean()
        assert return_network_output
        return raw


def test_prediction_is_known_skip_plus_network_and_protection_is_exact():
    x = jnp.full((2, 32, 32, 1), 0.4)
    s = jnp.full_like(x, 0.7)
    t = jnp.full_like(x, 0.5).at[:, :, :16].set(0)
    u = jnp.full_like(x, 0.3)
    m = ConstantNetwork()
    v, y = recursive_velocity(m, (x,), (t,), (s,), (u,))
    _, skip, cout = coefficients(t, u)
    np.testing.assert_allclose(v[0], jnp.where(t > 0, skip * (x - s) + cout * 0.2, 0), atol=2e-7)
    np.testing.assert_allclose(y[0], jnp.where(t > 0, x + u * v[0], s), atol=2e-7)
    np.testing.assert_array_equal(np.asarray(y[0])[:, :, :16], np.asarray(s)[:, :, :16])
    p = dict(
        images=(x,), levels=(jnp.zeros_like(t),), guides=(s,), progress=(u,), velocity_target=-x
    )
    for normalized in (False, True):
        (loss, _), grad = nnx.value_and_grad(
            lambda model, normalized=normalized: objective(model, p, normalized=normalized),
            has_aux=True,
        )(m)
        assert float(loss) == 0
        np.testing.assert_array_equal(grad["value"], 0)


def test_normalized_loss_is_physical_error_divided_by_cout_squared():
    x = jnp.full((2, 32, 32, 1), 0.4)
    s = jnp.full_like(x, 0.7)
    t = jnp.full_like(x, 0.5)
    u = jnp.full_like(x, 0.3)
    p = dict(images=(x,), levels=(t,), guides=(s,), progress=(u,), velocity_target=-x)
    physical = objective(ConstantNetwork(), p)[0]
    normalized = objective(ConstantNetwork(), p, normalized=True)[0]
    cout = float(coefficients(t, u)[2][0, 0, 0, 0])
    # Separate float32 reductions over2048 identical values differ by<2e-7.
    np.testing.assert_allclose(normalized, physical / cout**2, rtol=2e-5, atol=3e-7)


def test_real_updates_recurse_and_restore_both_independent_arms(tmp_path):
    _, p, _ = fixture(tmp_path)
    c = load_config(tmp_path / "configs/velocity.json")
    for name in ("physical", "normalized"):
        save_config(c, tmp_path / f"configs/{name}.json")
    arms = make_arms(tmp_path)
    before = jax.tree.map(
        lambda x: np.asarray(x).copy(), lineart_state.payload({"x": arms["normalized"]}, 0)
    )
    a = update(arms["physical"], p)
    jax.tree.map(
        np.testing.assert_array_equal, before, lineart_state.payload({"x": arms["normalized"]}, 0)
    )
    b = update(arms["normalized"], p)
    assert a["gradient_norm"] > 0 and b["gradient_norm"] > 0
    assert np.isfinite(a["loss"]) and np.isfinite(b["loss"])
    with checkpoints.manager(tmp_path / "checkpoints") as saved:
        lineart_state.save(saved, arms, 1)
        restored = make_arms(tmp_path)
        assert lineart_state.restore(saved, restored, 1) == 1
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 1),
        lineart_state.payload(restored, 1),
    )
    for name in arms:
        assert update(arms[name], p) == update(restored[name], p)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 2),
        lineart_state.payload(restored, 2),
    )


def test_raw_head_accessor_retains_default_physical_residual_output(tmp_path):
    model, p, _ = fixture(tmp_path)
    x, t, s, u = (p[k][0] for k in ("images", "levels", "guides", "progress"))
    insert = jnp.zeros((x.shape[0], x.shape[1] // 2, x.shape[2] // 2, 1), x.dtype)
    head = model(x, t, insert, s, u, return_network_output=True)
    clipped, raw = model(x, t, insert, s, u, return_preclip=True)
    np.testing.assert_array_equal(raw, x + jnp.sqrt(t) * head)
    np.testing.assert_array_equal(clipped, jnp.clip(raw, -1, 1))
    with pytest.raises(ValueError, match="cannot be combined"):
        model(x, t, insert, s, u, return_network_output=True, return_preclip=True)


def test_top_velocity_loss_reaches_distinct_smaller_stage_parameter():
    class TwoScale(nnx.Module):
        def __init__(self):
            self.small = nnx.Param(jnp.array(0.1))
            self.large = nnx.Param(jnp.array(0.2))

        def __call__(self, x, t, insert, s, u, return_network_output=False):
            assert return_network_output
            value = self.small[...] if x.shape[1] == 16 else self.large[...] + insert.mean()
            return jnp.full_like(x, value)

    images = (jnp.full((1, 32, 32, 1), 0.4), jnp.full((1, 16, 16, 1), 0.4))
    p = dict(
        images=images,
        guides=tuple(jnp.full_like(x, 0.7) for x in images),
        levels=tuple(jnp.full_like(x, 0.5) for x in images),
        progress=tuple(jnp.full_like(x, 0.3) for x in images),
        velocity_target=jnp.full_like(images[0], -0.9),
    )
    gradients = nnx.grad(lambda m: objective(m, p)[0])(TwoScale())
    assert abs(float(gradients["small"])) > 1e-6
    assert abs(float(gradients["large"])) > 1e-6
