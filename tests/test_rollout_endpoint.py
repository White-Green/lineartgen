import jax
import jax.numpy as jnp
import numpy as np
import pytest
from anchored_endpoint import endpoints, velocity
from diagnose_learnability import update
from flax import nnx
from rollout_endpoint import inputs, midpoint, objective
from rollout_endpoint_fit import run_sample, split_arms
from test_progress_modulation import modulated
from velocity_bridge_toy import initialize, path_inputs, sample

from lineartgen.training import checkpoints, lineart_state
from lineartgen.training.lineart_study import create_arm


class TwoTimes(nnx.Module):
    def __init__(self):
        self.early = nnx.Param(jnp.array(0.1))
        self.middle = nnx.Param(jnp.array(0.1))

    def __call__(self, x, t, insert, s, u, return_network_output=False):
        assert return_network_output
        return jnp.where(u > 0.75, self.early[...], self.middle[...])


def test_midpoint_gradient_matches_manual_two_time_derivative():
    x = jnp.zeros((1, 16, 16, 1))
    t = jnp.ones_like(x).at[:, :, :8].set(0)
    y = jnp.full_like(x, 0.5).at[:, :, :8].set(-1)
    p = dict(images=(x,), levels=(t,), guides=(x,), target=y)
    for teacher, expected_early, expected_middle in ((True, -0.4, -0.6), (False, -1.1, -1.4)):
        (loss, _), g = nnx.value_and_grad(
            lambda m, teach=teacher: objective(m, p, teach), has_aux=True
        )(TwoTimes())
        assert np.isfinite(loss)
        np.testing.assert_allclose(g["early"], expected_early, rtol=2e-6)
        np.testing.assert_allclose(g["middle"], expected_middle, rtol=2e-6)
        q = dict(p, levels=(jnp.zeros_like(t),))
        (loss, _), g = nnx.value_and_grad(
            lambda m, pp=q, teach=teacher: objective(m, pp, teach), has_aux=True
        )(TwoTimes())
        assert float(loss) == 0
        jax.tree.map(lambda a: np.testing.assert_array_equal(a, 0), g)


def test_rollout_midpoint_equals_euler_and_has_no_teacher_dependency(tmp_path):
    _, model, _, p = modulated(tmp_path)
    s, t = p["guides"][0], p["levels"][0]
    z, levels, guides = initialize(s, t, jax.random.PRNGKey(74), 2)
    direct = dict(
        images=z,
        levels=levels,
        guides=guides,
        progress=tuple(jnp.ones_like(x) for x in z),
        target=p["target"],
    )
    raw, bounded = endpoints(model, z, levels, guides, direct["progress"])
    v = velocity(model, z, levels, guides, direct["progress"])
    actual = midpoint(direct, bounded, teacher=False)
    for a, x, vel, level, source in zip(actual, z, v, levels, guides, strict=True):
        np.testing.assert_allclose(
            a, jnp.where(level > 0, x + 0.5 * vel, source), atol=2e-7, rtol=0
        )
        mask = np.asarray(level) == 0
        np.testing.assert_array_equal(np.asarray(a)[mask], np.asarray(source)[mask])
    changed = dict(direct, target=-direct["target"])
    jax.tree.map(np.testing.assert_array_equal, actual, midpoint(changed, bounded, teacher=False))
    assert (
        np.max(
            np.abs(
                np.asarray(midpoint(direct, bounded, teacher=True)[0])
                - np.asarray(midpoint(changed, bounded, teacher=True)[0])
            )
        )
        > 0.1
    )
    full = run_sample(model, s, t, jax.random.PRNGKey(74), 2, steps=2)
    nextu = tuple(jnp.full_like(x, 0.5) for x in actual)
    expected = endpoints(model, actual, levels, guides, nextu)[1][0]
    np.testing.assert_allclose(full, expected, atol=1e-6, rtol=0)
    direct_check = path_inputs(
        s, p["target"], t, jax.random.PRNGKey(74), jnp.zeros((s.shape[0], 1, 1, 1)), 2
    )
    jax.tree.map(np.testing.assert_array_equal, inputs({"direct": direct_check})["images"], z)


def test_actual_state_independence_restore_and_next_update(tmp_path):
    _, model, c, p = modulated(tmp_path)
    original = create_arm(c, model)
    configs = {m: c for m in ("teacher", "rollout")}
    arms = split_arms(original, configs)
    before = jax.tree.map(lambda x: np.asarray(x).copy(), lineart_state.payload({"x": original}, 0))
    for name in arms:
        stats = update(arms[name], p)
        assert stats["gradient_norm"] > 0 and np.isfinite(stats["loss"])
        jax.tree.map(
            np.testing.assert_array_equal, before, lineart_state.payload({"x": original}, 0)
        )
        if name == "teacher":
            jax.tree.map(
                np.testing.assert_array_equal,
                before,
                lineart_state.payload({"x": arms["rollout"]}, 0),
            )
    with checkpoints.manager(tmp_path / "state") as saved:
        lineart_state.save(saved, arms, 1)
        restored = split_arms(original, configs)
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


@pytest.mark.parametrize("steps", [2, 32])
def test_original_sampler_unchanged(tmp_path, steps):
    _, model, _, p = modulated(tmp_path)
    s, t = p["guides"][0], p["levels"][0]
    k = jax.random.PRNGKey(85)
    original = sample(
        lambda x, t, s, u: velocity(model, x, t, s, u, False), s, t, k, depth=2, steps=steps
    )
    np.testing.assert_array_equal(run_sample(model, s, t, k, 2, steps=steps), original)
