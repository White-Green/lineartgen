import jax
import jax.numpy as jnp
import numpy as np
import pytest
from anchored_endpoint import endpoints, objective, velocity
from anchored_endpoint_fit import make_arms
from diagnose_learnability import update
from flax import nnx
from learned_bridge_toy import endpoint as original_endpoint
from test_velocity_bridge_toy import fixture
from velocity_bridge_toy import batch, sample, velocity_inputs

from lineartgen.config import load_config, save_config
from lineartgen.training import checkpoints, lineart_state


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_noisy_endpoints_match_original_AT_at_all_scales(tmp_path, depth):
    model, _, _ = fixture(tmp_path)
    p = velocity_inputs(batch(4, depth, count=2)[0])
    args = {k: p[k] for k in ("images", "levels", "guides", "progress")}
    actual, bounded = endpoints(model, **args, source_anchor=False)
    expected = original_endpoint(model, **args)
    jax.tree.map(np.testing.assert_array_equal, actual, expected)
    jax.tree.map(
        lambda a, b: np.testing.assert_array_equal(a, jnp.clip(b, -1, 1)), bounded, expected
    )
    if depth == 1:
        shifted = endpoints(model, **args, source_anchor=True)[0][0]
        np.testing.assert_allclose(
            shifted - actual[0], p["guides"][0] - p["images"][0], atol=3e-7, rtol=0
        )


class ZeroHead(nnx.Module):
    def __call__(self, x, t, insert, s, u, return_network_output=False):
        assert return_network_output
        return jnp.zeros_like(x)


@pytest.mark.parametrize("steps", [1, 8, 32])
def test_zero_source_head_removes_sampled_noise_and_protects_source(steps):
    s = jnp.ones((2, 32, 32, 1)).at[:, 5:25, 14:18].set(0.4)
    t = jnp.full_like(s, 0.5).at[:, :, :16].set(0)
    result = sample(
        lambda x, t, s, u: velocity(ZeroHead(), x, t, s, u, True),
        s,
        t,
        jax.random.PRNGKey(8),
        depth=2,
        steps=steps,
    )
    np.testing.assert_allclose(result, s, atol=2e-7, rtol=0)
    np.testing.assert_array_equal(np.asarray(result)[:, :, :16], np.asarray(s)[:, :, :16])


def test_interval_endpoint_gradients_respect_AA_saturation_masks_and_time_cap():
    class Head(nnx.Module):
        def __init__(self):
            self.values = nnx.Param(jnp.array([1.2, -1.2, 0.3, 0, 0, 0]))

        def __call__(self, x, t, insert, s, u, return_network_output=False):
            return jnp.broadcast_to(self.values[...][None, None, :, None], x.shape)

    x = jnp.zeros((1, 16, 6, 1))
    y = jnp.broadcast_to(jnp.array([1, -1, 0.3, 1, -1, 0.3])[None, None, :, None], x.shape)
    t = jnp.ones_like(x)
    u = jnp.full_like(x, 1e-8)
    p = dict(images=(x,), levels=(t,), guides=(x,), progress=(u,), target=y)
    (loss, _), g = nnx.value_and_grad(lambda m: objective(m, p, True), has_aux=True)(Head())
    assert np.isfinite(float(loss)) and np.isfinite(np.asarray(g["values"])).all()
    np.testing.assert_array_equal(g["values"][:3], 0)
    assert float(g["values"][3]) < 0 and float(g["values"][4]) > 0 and float(g["values"][5]) < 0
    p["levels"] = (t.at[:, :, 3:].set(0),)
    value, g = nnx.value_and_grad(lambda m: objective(m, p, True)[0])(Head())
    assert float(value) == 0
    np.testing.assert_array_equal(g["values"], 0)


def test_actual_matched_updates_and_full_state_next_update_restore(tmp_path):
    _, p, _ = fixture(tmp_path)
    c = load_config(tmp_path / "configs/velocity.json")
    for name in ("noisy", "source"):
        save_config(c, tmp_path / f"configs/{name}.json")
    arms = make_arms(tmp_path)
    before = jax.tree.map(
        lambda x: np.asarray(x).copy(), lineart_state.payload({"x": arms["source"]}, 0)
    )
    a = update(arms["noisy"], p)
    jax.tree.map(
        np.testing.assert_array_equal, before, lineart_state.payload({"x": arms["source"]}, 0)
    )
    b = update(arms["source"], p)
    assert a["gradient_norm"] > 0 and b["gradient_norm"] > 0 and a["loss"] != b["loss"]
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


def test_larger_endpoint_loss_reaches_separate_smaller_stage():
    class TwoScale(nnx.Module):
        def __init__(self):
            self.small = nnx.Param(jnp.array(0.1))
            self.large = nnx.Param(jnp.array(0.2))

        def __call__(self, x, t, insert, s, u, return_network_output=False):
            value = self.small[...] if x.shape[1] == 16 else self.large[...] + insert.mean()
            return jnp.full_like(x, value)

    xs = (jnp.zeros((1, 32, 32, 1)), jnp.zeros((1, 16, 16, 1)))
    p = dict(
        images=xs,
        guides=xs,
        levels=tuple(jnp.full_like(x, 0.5) for x in xs),
        progress=tuple(jnp.full_like(x, 0.5) for x in xs),
        target=jnp.full_like(xs[0], 0.5),
    )
    for source_anchor in (False, True):
        g = nnx.grad(lambda m, a=source_anchor: objective(m, p, a)[0])(TwoScale())
        assert abs(float(g["small"])) > 1e-6 and abs(float(g["large"])) > 1e-6
