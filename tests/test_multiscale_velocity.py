import jax
import jax.numpy as jnp
import numpy as np
import pytest
from diagnose_learnability import update
from flax import nnx
from geometry_velocity_fit import prepared_batch
from multiscale_velocity_fit import make_arms
from multiscale_velocity_objective import (
    multiscale_inputs,
    multiscale_objective,
    stage_velocity_mse,
)
from test_velocity_bridge_toy import TwoStage, fixture
from velocity_bridge_toy import batch, objective, velocity_inputs

from lineartgen.config import load_config, save_config
from lineartgen.diffusion import pyramid
from lineartgen.training import checkpoints, lineart_state


@pytest.mark.parametrize("depth", [1, 6])
def test_each_stage_target_uses_its_own_initial_noise_and_path_derivative(depth):
    y = jnp.ones((2, 512, 512, 1)).at[:, 100:300, 245:248].set(-0.5)
    s = 1 - 0.3 * (1 - y)
    t = jnp.full_like(y, 0.5).at[:, :, :256].set(0)
    p, tau = prepared_batch(y, s, t, 1, depth)
    q = multiscale_inputs(p)
    old = velocity_inputs(p)
    jax.tree.map(np.testing.assert_array_equal, old, {k: q[k] for k in old})
    clean = pyramid(jnp.where(t > 0, y, s), depth)
    for k, target in enumerate(q["velocity_targets"]):
        np.testing.assert_array_equal(target, clean[k] - p["direct"]["images"][k])
        expected = p["direct"]["images"][k] + tau * target
        np.testing.assert_allclose(q["images"][k], expected, atol=3e-7, rtol=0)
        np.testing.assert_array_equal(np.asarray(target)[np.asarray(q["levels"][k]) == 0], 0)
    if depth > 1:
        assert not np.allclose(q["velocity_targets"][1], pyramid(q["velocity_targets"][0], 2)[1])


def test_zero_weight_and_single_stage_match_original_loss_and_gradient(tmp_path):
    model, _, _ = fixture(tmp_path)
    for depth, multiplier in ((2, 0.0), (1, 1.0)):
        p = multiscale_inputs(batch(2, depth, count=2)[0])
        a, ga = nnx.value_and_grad(lambda m, p=p: objective(m, p)[0])(model)
        b, gb = nnx.value_and_grad(
            lambda m, p=p, multiplier=multiplier: multiscale_objective(m, p, multiplier)[0]
        )(model)
        np.testing.assert_array_equal(a, b)
        jax.tree.map(np.testing.assert_array_equal, ga, gb)


def test_every_stage_receives_direct_gradient_and_recursion_gradient():
    model = TwoStage()
    images = (jnp.zeros((1, 32, 32, 1)), jnp.zeros((1, 16, 16, 1)))
    levels = tuple(jnp.full_like(x, 0.5) for x in images)
    p = dict(
        images=images,
        levels=levels,
        guides=images,
        progress=levels,
        velocity_target=jnp.full_like(images[0], 0.7),
        velocity_targets=tuple(jnp.full_like(x, 0.7) for x in images),
    )
    value, g = nnx.value_and_grad(lambda m: multiscale_objective(m, p)[0])(model)
    np.testing.assert_allclose(value, (0.45**2 + 0.6**2) / 2, atol=1e-6)
    np.testing.assert_allclose(g["small"], -0.825, atol=1e-5)
    np.testing.assert_allclose(g["large"], -0.45, atol=1e-5)
    protected = dict(p, levels=tuple(jnp.zeros_like(x) for x in images))
    value, g = nnx.value_and_grad(lambda m: multiscale_objective(m, protected)[0])(model)
    assert value == 0
    assert all(np.all(x == 0) for x in jax.tree.leaves(g))


def test_per_image_active_normalization_and_protected_gradients():
    p = jnp.full((2, 4, 4, 1), 2.0)
    t = jnp.ones_like(p).at[0, :3].set(0).at[1].set(0)
    v, g = jax.value_and_grad(stage_velocity_mse)(p, jnp.zeros_like(p), t)
    assert v == 2
    assert np.isfinite(g).all()
    np.testing.assert_array_equal(np.asarray(g)[np.asarray(t) == 0], 0)


def test_actual_multiscale_update_and_full_restore(tmp_path):
    fixture(tmp_path)
    c = load_config(tmp_path / "configs/velocity.json")
    c = c.model_copy(
        update={"diffusion": c.diffusion.model_copy(update={"scale_loss_multiplier": 1.0})}
    )
    save_config(c, tmp_path / "configs/multiscale.json")
    p = multiscale_inputs(batch(2, 2, count=2)[0])
    arms = make_arms(tmp_path)
    m = update(arms["multiscale"], p)
    assert m["gradient_norm"] > 0 and m["velocity_mse_scale1"] > 0
    with checkpoints.manager(tmp_path / "checkpoints") as saved:
        lineart_state.save(saved, arms, 1)
        restored = make_arms(tmp_path)
        assert lineart_state.restore(saved, restored) == 1
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 1),
        lineart_state.payload(restored, 1),
    )
    assert update(arms["multiscale"], p) == update(restored["multiscale"], p)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 2),
        lineart_state.payload(restored, 2),
    )
