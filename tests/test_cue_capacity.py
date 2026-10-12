import jax
import jax.numpy as jnp
import numpy as np
from cue_capacity import forward, make_arm, prepare_inputs
from cue_preserving_data import corrupt, inverse, make_case, strength_map, transfer
from diagnose_learnability import update
from flax import nnx
from local_recursive_fit import fresh_model, local_config

from lineartgen.config import ExperimentConfig, LossConfig, NoiseMatchingConfig, TrainConfig
from lineartgen.diffusion import pyramid
from lineartgen.training import checkpoints, lineart_state
from lineartgen.training.losses import image_loss


def configuration(width):
    return ExperimentConfig(
        matching=NoiseMatchingConfig(method="independent"),
        model=local_config().model_copy(update={"channels": (width,)}),
        loss=LossConfig(
            pixel_loss="interval_l1", edge_weight=0, contrast_weight=0, noise_weight_floor=0.05
        ),
        train=TrainConfig(
            steps=20, warmup_steps=0, batch_size=1, accumulate_steps=1, model_seed=42
        ),
    )


def test_information_retained_and_shapes_disjoint():
    hashes = {}
    for split in ("train", "probe"):
        values = []
        for index in range(96):
            data, record = make_case(split, index)
            assert record["inverse_max_error"] < 2e-6
            assert np.min(data["source"]) >= -1 and np.max(data["source"]) <= 1
            if record["kind"] not in ("blank", "black"):
                values.append(record["clean_sha256"])
        assert len(set(values)) == 88
        hashes[split] = set(values)
    assert hashes["train"].isdisjoint(hashes["probe"])
    assert np.min(transfer((256, 256))) >= 0.4
    # Thin and wide targets must remain distinguishable, without clipping the source.
    a = np.ones((64, 64, 1), np.float32)
    a[32] = -1
    b = a.copy()
    b[30:35] = -1
    assert np.max(np.abs(corrupt(a) - corrupt(b))) > 0.1
    np.testing.assert_allclose(inverse(corrupt(a)), a, atol=2e-6)


def test_noise_after_reduction_and_no_teacher_input():
    source = jnp.linspace(-0.6, 1, 64**2).reshape(1, 64, 64, 1)
    strength = jnp.asarray(strength_map("partial05", 64))[None]
    key = jax.random.PRNGKey(17)
    p = prepare_inputs(-source, source, strength, key, depth=4)
    different = prepare_inputs(jnp.zeros_like(source), source, strength, key, depth=4)
    jax.tree.map(np.testing.assert_array_equal, p[1:], different[1:])
    for s, t, x, k in zip(
        pyramid(source, 4), pyramid(strength, 4), p.noisy, jax.random.split(key, 4), strict=True
    ):
        expected = s + 0.25 * t * jax.random.normal(k, s.shape)
        np.testing.assert_allclose(x, expected, atol=2e-7)
        np.testing.assert_array_equal(
            np.asarray(x)[np.asarray(t) == 0], np.asarray(s)[np.asarray(t) == 0]
        )
    assert np.max(np.abs(np.asarray(pyramid(p.noisy[0], 4)[1] - p.noisy[1]))) > 0.01
    model = fresh_model(configuration(16))
    a, b = forward(model, p), forward(model, different)
    jax.tree.map(np.testing.assert_array_equal, a, b)


def test_all_stage_losses_and_independent_updates(tmp_path):
    c = configuration(16)
    c = c.model_copy(
        update={"diffusion": c.diffusion.model_copy(update={"scale_loss_multiplier": 0.5})}
    )
    source = jnp.ones((1, 32, 32, 1)).at[:, 8:24, 15:17].set(0.4)
    clean = jnp.where(source < 1, -1.0, 1.0)
    p = prepare_inputs(clean, source, jnp.full_like(source, 0.5), jax.random.PRNGKey(1), depth=4)
    small = make_arm(c, fresh_model(c))
    wide_c = c.model_copy(update={"model": local_config().model_copy(update={"channels": (32,)})})
    wide = make_arm(wide_c, fresh_model(wide_c))
    jax.tree.map(
        np.testing.assert_array_equal, forward(small["model"], p), forward(wide["model"], p)
    )
    predictions, raw = forward(small["model"], p)
    losses = [
        image_loss(a, b, t, c.loss, c.diffusion.min_noise, preclip=z)[0]
        for a, b, t, z in zip(predictions, p.targets, p.noise_levels, raw, strict=True)
    ]
    expected = sum(0.5**i * v for i, v in enumerate(losses)) / sum(0.5**i for i in range(4))
    # Fused JIT and separate float32 reductions have slightly different order.
    np.testing.assert_allclose(small["measure"](small["model"], p)["loss"], expected, rtol=2e-5)
    assert abs(float(expected - losses[0])) > 1e-3
    state_before = nnx.to_pure_dict(nnx.state(wide["model"], nnx.Param))
    assert update(small, p)["gradient_norm"] > 0
    jax.tree.map(
        np.testing.assert_array_equal,
        state_before,
        nnx.to_pure_dict(nnx.state(wide["model"], nnx.Param)),
    )
    assert update(wide, p)["gradient_norm"] > 0
    arms = dict(width16=small, width32=wide)
    with checkpoints.manager(tmp_path / "checkpoint") as manager:
        lineart_state.save(manager, arms, 1)
        restored = {
            "width16": make_arm(c, fresh_model(c)),
            "width32": make_arm(wide_c, fresh_model(wide_c)),
        }
        assert lineart_state.restore(manager, restored) == 1
    for name in arms:
        assert update(arms[name], p) == update(restored[name], p)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 2),
        lineart_state.payload(restored, 2),
    )
