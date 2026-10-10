import jax
import jax.numpy as jnp
import numpy as np
import pytest
from diagnose_learnability import update
from flax import nnx
from learned_bridge_toy import batch as old_batch
from learned_bridge_toy import endpoint as old_endpoint
from velocity_bridge_toy import (
    batch,
    config,
    endpoint,
    make_arms,
    path_inputs,
    recursive_velocity,
    sample,
    velocity_inputs,
)

from lineartgen.config import ModelConfig, save_config
from lineartgen.models import create_model
from lineartgen.models.serialization import save_inference
from lineartgen.training import checkpoints, lineart_state


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_all_prepared_data_matches_AT_and_velocity_target_is_path_derivative(depth):
    for step in (0, 1, 1999, 5999):
        actual = batch(step, depth)
        expected = old_batch(step, depth)
        jax.tree.map(np.testing.assert_array_equal, actual, expected)
        values = velocity_inputs(actual[0])
        p = actual[0]
        np.testing.assert_array_equal(
            values["velocity_target"], p["bridge"]["target"] - p["direct"]["images"][0]
        )
        mask = np.asarray(values["levels"][0]) == 0
        np.testing.assert_array_equal(np.asarray(values["velocity_target"])[mask], 0)


@pytest.mark.parametrize("steps", [1, 32, 256])
def test_bounded_velocity_has_finite_endpoint_without_remaining_time_division(steps):
    source = jnp.ones((2, 32, 32, 1))
    strength = jnp.full_like(source, 0.5).at[:, :, :16].set(0)
    key = jax.random.PRNGKey(3)
    from velocity_bridge_toy import initialize

    initial = initialize(source, strength, key, 2)[0][0]
    result = sample(
        lambda x, t, s, u: tuple(jnp.full_like(v, 0.25) for v in x),
        source,
        strength,
        key,
        depth=2,
        steps=steps,
    )
    np.testing.assert_allclose(
        result, jnp.where(strength > 0, initial + 0.25, source), atol=2e-5, rtol=0
    )
    np.testing.assert_array_equal(np.asarray(result)[:, :, :16], 1)


class TwoStage(nnx.Module):
    def __init__(self):
        self.small = nnx.Param(jnp.array(0.1))
        self.large = nnx.Param(jnp.array(0.2))

    def __call__(self, x, t, insert, s, u, return_preclip=False):
        v = self.small[...] if x.shape[1] == 16 else self.large[...] + insert.mean()
        raw = x + v
        return jnp.clip(raw, -1, 1), raw


def test_recursive_insert_is_time_correct_and_gradient_reaches_smaller_stage():
    model = TwoStage()
    images = (jnp.zeros((1, 32, 32, 1)), jnp.zeros((1, 16, 16, 1)))
    levels = tuple(jnp.full_like(x, 0.5) for x in images)
    progress = tuple(jnp.full_like(x, 0.5) for x in images)
    v, y = recursive_velocity(model, images, levels, images, progress)
    np.testing.assert_allclose(v[0], 0.25, atol=1e-6)
    np.testing.assert_allclose(y[0], 0.125, atol=1e-6)
    np.testing.assert_allclose(y[1], 0.05, atol=1e-6)
    grads = nnx.grad(
        lambda m: jnp.mean(
            (recursive_velocity(m, images, levels, images, progress)[0][0] - 0.7) ** 2
        )
    )(model)
    # Float32 reduction over 1024 pixels incurs a few e-6 of accumulation error.
    np.testing.assert_allclose(grads["small"], -0.45, atol=1e-5, rtol=0)
    zero = tuple(jnp.zeros_like(x) for x in images)
    jax.tree.map(
        np.testing.assert_array_equal, endpoint(model, images, levels, images, zero), images
    )


def fixture(root):
    c = config().model_copy(
        update={
            "model": ModelConfig(
                channels=(2, 4),
                convolutions_per_block=2,
                detail_channels=4,
                output_mode="residual",
                source_conditioning=True,
                progress_conditioning=True,
            ),
            "train": config().train.model_copy(update={"batch_size": 2, "warmup_steps": 0}),
        }
    )
    model = create_model(c.model, rngs=nnx.Rngs(42))
    save_inference(model, c, root / "initial", provenance="test")
    (root / "configs").mkdir()
    save_config(c, root / "configs/velocity.json")
    p, _, _ = batch(1, 2, count=2)
    q = p["direct"]
    p["bridge"] = path_inputs(
        q["guides"][0],
        q["target"],
        q["levels"][0],
        jax.random.PRNGKey(19),
        jnp.full((2, 1, 1, 1), 0.5),
        2,
    )
    return model, velocity_inputs(p), p["direct"]


def test_initial_time_matches_original_model_and_real_update_restores(tmp_path):
    model, prepared, direct = fixture(tmp_path)
    args = {k: direct[k] for k in ("images", "levels", "guides", "progress")}
    actual = endpoint(model, **args)
    expected = old_endpoint(model, **args)
    jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b, atol=1e-5, rtol=0), actual, expected)
    arms = make_arms(tmp_path)
    before = jax.tree.map(lambda x: np.asarray(x).copy(), nnx.state(model, nnx.Param))
    result = update(arms["velocity"], prepared)
    assert result["gradient_norm"] > 0 and np.isfinite(result["loss"])
    after = nnx.state(arms["velocity"]["model"], nnx.Param)
    assert any(
        np.any(a != b) for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(after), strict=True)
    )
    with checkpoints.manager(tmp_path / "checkpoints") as saved:
        lineart_state.save(saved, arms, 1)
        restored = make_arms(tmp_path)
        assert lineart_state.restore(saved, restored) == 1
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 1),
        lineart_state.payload(restored, 1),
    )
    assert update(arms["velocity"], prepared) == update(restored["velocity"], prepared)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 2),
        lineart_state.payload(restored, 2),
    )
