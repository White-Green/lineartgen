import jax
import jax.numpy as jnp
import numpy as np
from diagnose_learnability import update
from distance_supervision_fit import make_arms, objective, prepared_batch
from flax import nnx
from paper_band_fit import objective as base_objective

from lineartgen.config import ExperimentConfig, LossConfig, ModelConfig, TrainConfig, save_config
from lineartgen.models import create_model
from lineartgen.models.serialization import save_inference
from lineartgen.training import checkpoints, lineart_state
from lineartgen.training.coverage_distance import coverage_distance
from lineartgen.training.distance_supervision import distance_supervision


def sample_target():
    y = np.ones((32, 32, 1), np.float32)
    y[8:24, 15:17] = -1
    y[8:24, 14:15] = 0.4
    return jnp.asarray(y[None]), jnp.asarray(coverage_distance(y)[None])


def test_correct_field_and_cap_saturation_have_zero_loss_and_gradient():
    y, d = sample_target()
    for p in (2 * d, jnp.where(d == 8, 40, jnp.where(d == -8, -40, 2 * d))):
        value, gradient = jax.value_and_grad(distance_supervision)(p, d, y, jnp.ones_like(y, bool))
        assert value == 0
        np.testing.assert_array_equal(gradient, 0)


def test_wrong_field_has_inward_gradients_and_actual_descent():
    y, d = sample_target()
    p = -2 * d
    active = jnp.ones_like(y, bool)
    value, gradient = jax.value_and_grad(distance_supervision)(p, d, y, active)
    assert value > 0 and np.isfinite(gradient).all()
    assert np.all(np.asarray(gradient)[np.asarray(d) > 0] < 0)
    assert np.all(np.asarray(gradient)[np.asarray(d) < 0] > 0)
    assert distance_supervision(p - gradient, d, y, active) < value


def test_protected_region_and_all_protected_are_zero_gradient():
    y, d = sample_target()
    mask = jnp.ones_like(y, bool).at[:, :, :16].set(False)
    _, g = jax.value_and_grad(distance_supervision)(-2 * d, d, y, mask)
    np.testing.assert_array_equal(g[:, :, :16], 0)
    v, g = jax.value_and_grad(distance_supervision)(-2 * d, d, y, jnp.zeros_like(mask))
    assert v == 0
    np.testing.assert_array_equal(g, 0)


def fixture(root):
    config = ExperimentConfig(
        model=ModelConfig(
            channels=(2, 4),
            convolutions_per_block=2,
            detail_channels=4,
            source_conditioning=True,
            feature_input="source",
            residual_base="source",
            sampling_mode="one_pass",
            output_mode="residual",
        ),
        loss=LossConfig(pixel_loss="interval_l1", tone_balance_power=0.5),
        train=TrainConfig(batch_size=2, accumulate_steps=1, warmup_steps=0),
    )
    model = create_model(config.model, rngs=nnx.Rngs(42))
    save_inference(model, config, root / "initial", provenance="test")
    (root / "configs").mkdir()
    save_config(config, root / "configs/distance.json")
    y, _ = sample_target()
    y = jnp.repeat(y, 2, axis=0)
    p = prepared_batch(y, 1 - 0.3 * (1 - y), jnp.full_like(y, 0.5), 2)
    return model, config, p


def test_zero_weight_matches_base_loss_and_all_parameter_gradients(tmp_path):
    model, config, p = fixture(tmp_path)
    (a, _), ga = nnx.value_and_grad(lambda m: objective(m, p, config, weight=0), has_aux=True)(
        model
    )
    (b, _), gb = nnx.value_and_grad(lambda m: base_objective(m, p, config), has_aux=True)(model)
    assert np.isfinite(a) and np.isfinite(b)
    np.testing.assert_array_equal(a, b)
    jax.tree.map(np.testing.assert_array_equal, ga, gb)


def test_actual_update_and_optimizer_ema_restore(tmp_path):
    _, _, p = fixture(tmp_path)
    arms = make_arms(tmp_path)
    m = update(arms["distance"], p)
    assert m["gradient_norm"] > 0 and m["distance_supervision"] > 0
    with checkpoints.manager(tmp_path / "checkpoints") as saved:
        lineart_state.save(saved, arms, 1)
        restored = make_arms(tmp_path)
        assert lineart_state.restore(saved, restored) == 1
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 1),
        lineart_state.payload(restored, 1),
    )
    assert update(arms["distance"], p) == update(restored["distance"], p)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 2),
        lineart_state.payload(restored, 2),
    )
