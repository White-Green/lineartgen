import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from campaign_protocol import write
from diagnose_learnability import update
from edge_energy_fit import make_arms, prepared_inputs
from flax import nnx
from train_procedural_structure import prepare_batch

from lineartgen.config import (
    DiffusionConfig,
    ExperimentConfig,
    LossConfig,
    ModelConfig,
    TrainConfig,
    save_config,
)
from lineartgen.models import create_model
from lineartgen.models.serialization import save_inference
from lineartgen.training import checkpoints, lineart_state
from lineartgen.training.edge_energy import edge_energy_loss
from lineartgen.training.losses import image_loss


def target():
    return jnp.ones((2, 32, 32, 1)).at[:, 8:24, 10:12].set(-1).at[:, 8:24, 18:20].set(-0.3)


def test_correct_aa_and_valid_saturation_have_zero_loss_and_gradient():
    y = target()
    z = jnp.where(y == 1, 2.5, jnp.where(y == -1, -2.5, y))
    for p in (y, z):
        value, gradient = jax.value_and_grad(edge_energy_loss)(p, y, jnp.ones_like(y, bool))
        assert value == 0
        np.testing.assert_array_equal(gradient, 0)


def test_wrong_saturation_has_inward_gradient_and_actual_descent():
    y = target()
    z = jnp.where(y < 1, 1.5, y)
    active = jnp.ones_like(y, bool)
    value, gradient = jax.value_and_grad(edge_energy_loss)(z, y, active)
    assert value > 0 and np.isfinite(gradient).all()
    assert float(gradient[:, 8, 10:12].sum()) > 0
    assert edge_energy_loss(z - gradient, y, active) < value


def test_spatial_error_is_padding_and_rotation_invariant_with_fixed_edge_content():
    y = target()
    z = 1 - (1 - y) * 0.6
    active = jnp.ones_like(y, bool)
    expected = edge_energy_loss(z, y, active)
    for turns in (1, 2, 3):
        actual = edge_energy_loss(jnp.rot90(z, turns, (1, 2)), jnp.rot90(y, turns, (1, 2)), active)
        np.testing.assert_allclose(actual, expected, rtol=1e-6)
    pads = ((0, 0), (12, 12), (12, 12), (0, 0))
    actual = edge_energy_loss(
        jnp.pad(z, pads, constant_values=1),
        jnp.pad(y, pads, constant_values=1),
        jnp.pad(active, pads, constant_values=True),
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


@pytest.mark.parametrize("value", [-1.0, 0.2, 1.0])
def test_uniform_residual_nullspace_and_all_protected_are_finite(value):
    y = jnp.full((2, 32, 32, 1), value)
    for active in (jnp.ones_like(y, bool), jnp.zeros_like(y, bool)):
        loss, gradient = jax.value_and_grad(edge_energy_loss)(-y, y, active)
        assert loss == 0
        np.testing.assert_array_equal(gradient, 0)
    damaged = y.at[:, 12, 12].set(0.7 if value != 0.7 else 0)
    assert edge_energy_loss(damaged, y, jnp.ones_like(y, bool)) > 0


def test_mask_excludes_protected_gradients_and_stops_teacher_gradient():
    y = target()
    z = y * 0.5
    active = jnp.ones_like(y, bool).at[:, :, :16].set(False)
    gradient = jax.grad(edge_energy_loss)(z, y, active)
    np.testing.assert_array_equal(np.asarray(gradient)[:, :, :16], 0)
    assert np.any(np.asarray(gradient)[:, :, 16:] != 0)
    np.testing.assert_array_equal(jax.grad(edge_energy_loss, argnums=1)(z, y, active), 0)
    altered = z.at[:, :, :16].set(100)
    np.testing.assert_array_equal(
        edge_energy_loss(altered, y, active), edge_energy_loss(z, y, active)
    )


def test_disabled_loss_and_gradient_match_frozen_AR_exactly():
    path = Path("../jax-full-width-paper-20261010/src/lineartgen/training/losses.py").resolve()
    spec = importlib.util.spec_from_file_location("lineartgen.training.frozen_ar_losses", path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    y = target()
    z = y * 0.5 + 0.2
    t = jnp.full_like(y, 0.5)
    config = LossConfig(
        pixel_loss="interval_l1",
        paper_band_weight=0.1,
        gap_clearance_weight=0.1,
        opacity_deficit_weight=0.3,
        opacity_deficit_preclip=True,
        contrast_active_only=True,
        tone_balance_power=0.5,
    )

    def objective(fn, p):
        return fn(jnp.clip(p, -1, 1), y, t, config, 0.001, preclip=p)[0]

    jax.tree.map(
        np.testing.assert_array_equal,
        jax.value_and_grad(lambda p: objective(image_loss, p))(z),
        jax.value_and_grad(lambda p: objective(old.image_loss, p))(z),
    )


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_weight_rejected(value):
    with pytest.raises(ValueError):
        LossConfig(edge_energy_weight=value)


def test_missing_physical_prediction_and_unsupported_model_rejected():
    y = target()
    with pytest.raises(ValueError, match="physical preclip"):
        image_loss(y, y, jnp.ones_like(y), LossConfig(edge_energy_weight=0.1), 0.001)
    with pytest.raises(ValueError, match="single UNet"):
        ExperimentConfig(
            model=ModelConfig(output_clip_gradient="straight_through"),
            loss=LossConfig(edge_energy_weight=0.1),
        )


def fixture(root):
    config = ExperimentConfig(
        model=ModelConfig(
            channels=(2, 4),
            convolutions_per_block=2,
            detail_channels=4,
            output_mode="residual",
            source_conditioning=True,
            residual_base="source",
            feature_input="source",
            sampling_mode="one_pass",
        ),
        diffusion=DiffusionConfig(scale_loss_multiplier=0),
        loss=LossConfig(
            pixel_loss="interval_l1",
            edge_energy_weight=0.1,
            gap_clearance_weight=0.1,
            opacity_deficit_weight=0.3,
            opacity_deficit_preclip=True,
        ),
        train=TrainConfig(steps=10, batch_size=2, accumulate_steps=1, warmup_steps=0),
    )
    initial = create_model(config.model, rngs=nnx.Rngs(42))
    save_inference(initial, config, root / "initial", provenance="test fixture")
    (root / "configs").mkdir()
    save_config(config, root / "configs/energy.json")
    write(root / "plan.json", dict(initial=str(root / "initial")))
    y = np.asarray(target())
    source = np.ones_like(y)
    source[:, 8:24, 10:20] = -0.2
    return config, prepared_inputs(y, source, np.full_like(y, 0.5), 2)


def test_actual_gap_update_full_state_and_next_update_restore(tmp_path):
    _, prepared = fixture(tmp_path)
    arms = make_arms(tmp_path)
    before = jax.tree.map(np.asarray, nnx.state(arms["energy"]["model"], nnx.Param))
    metrics = update(arms["energy"], prepared)
    assert metrics["edge_energy"] > 0 and metrics["gradient_norm"] > 0
    assert any(
        np.any(a != b)
        for a, b in zip(
            jax.tree.leaves(before),
            jax.tree.leaves(nnx.state(arms["energy"]["model"], nnx.Param)),
            strict=True,
        )
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
    assert update(arms["energy"], prepared) == update(restored["energy"], prepared)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 2),
        lineart_state.payload(restored, 2),
    )


def test_generic_trainers_compute_physical_values_for_gap_only_loss(tmp_path):
    from lineartgen.training.lineart_steps import objective
    from lineartgen.training.steps import make_steps

    config, _ = fixture(tmp_path)
    config = config.model_copy(update={"loss": LossConfig(edge_energy_weight=0.1)})
    prepared = prepare_batch(config)(target(), jax.random.PRNGKey(4), 2)
    model = create_model(config.model, rngs=nnx.Rngs(42))
    value, parts, _ = objective(model, prepared, None, config)
    assert np.isfinite(float(value)) and "edge_energy" in parts
    metrics, grads = make_steps(config)[0](model, prepared)
    assert np.isfinite(float(metrics["loss"])) and "edge_energy" in metrics
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(grads))
