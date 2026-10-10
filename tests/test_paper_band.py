import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from campaign_protocol import write
from diagnose_learnability import update
from flax import nnx
from paper_band_fit import make_arms, prepared_inputs
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
from lineartgen.training.losses import image_loss
from lineartgen.training.paper_band import paper_band_importance, paper_band_loss


def target():
    a = jnp.ones((2, 32, 32, 1))
    return a.at[:, 8:24, 10:12].set(-1).at[:, 8:24, 18:20].set(-1)


def test_clearance_identifies_parallel_paper_and_has_inward_saturation_gradient():
    y = target()
    active = jnp.ones_like(y, bool)
    weights = paper_band_importance(y, active)
    assert float(weights[:, 10:22, 12:18].sum()) > 0
    np.testing.assert_array_equal(np.asarray(weights)[np.asarray(y) < 0.96], 0)
    z = jnp.where(weights > 0, -1.5, y)
    loss, gradient = jax.value_and_grad(paper_band_loss)(z, y, active)
    assert float(loss) > 0
    assert np.all(np.asarray(gradient)[np.asarray(weights) > 0] < 0)
    np.testing.assert_array_equal(np.asarray(gradient)[np.asarray(weights) == 0], 0)
    value, grad = jax.value_and_grad(paper_band_loss)(y, y, active)
    assert value == 0
    np.testing.assert_array_equal(grad, 0)
    np.testing.assert_array_equal(
        paper_band_importance(jnp.rot90(y, axes=(1, 2)), active), jnp.rot90(weights, axes=(1, 2))
    )


@pytest.mark.parametrize("value", [-1.0, 1.0])
def test_uniform_fields_have_zero_finite_loss_and_gradient(value):
    y = jnp.full((2, 32, 32, 1), value)
    loss, grad = jax.value_and_grad(paper_band_loss)(-y, y, jnp.ones_like(y, bool))
    assert loss == 0
    np.testing.assert_array_equal(grad, 0)


def test_protected_pixels_are_excluded_and_isolated_straight_fringe_is_selected():
    y = target()
    loss, grad = jax.value_and_grad(paper_band_loss)(-y, y, jnp.zeros_like(y, bool))
    assert loss == 0
    np.testing.assert_array_equal(grad, 0)
    single = jnp.ones_like(y).at[:, 8:24, 14:17].set(-1)
    assert float(paper_band_importance(single, jnp.ones_like(y, bool)).sum()) > 0


def test_default_loss_and_gradient_exactly_match_frozen_AA():
    path = Path(
        "/home/whitegreen/projects/lineartgen/.worktrees/jax-gap-clearance-fit-20261010/src/lineartgen/training/losses.py"
    )
    spec = importlib.util.spec_from_file_location("lineartgen.training.frozen_aa_losses", path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    y = target()
    z = y * 0.5 + 0.2
    t = jnp.full_like(y, 0.5)
    config = LossConfig(
        pixel_loss="interval_l1",
        gap_clearance_weight=0.1,
        opacity_deficit_weight=0.3,
        opacity_deficit_preclip=True,
        contrast_active_only=True,
        tone_balance_power=0.5,
    )

    def objective(fn, z):
        return fn(jnp.clip(z, -1, 1), y, t, config, 0.001, preclip=z)[0]

    actual = jax.value_and_grad(lambda z: objective(image_loss, z))(z)
    expected = jax.value_and_grad(lambda z: objective(old.image_loss, z))(z)
    jax.tree.map(np.testing.assert_array_equal, actual, expected)


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
            paper_band_weight=0.1,
            gap_clearance_weight=0.1,
            opacity_deficit_weight=0.3,
            opacity_deficit_preclip=True,
        ),
        train=TrainConfig(steps=10, batch_size=2, accumulate_steps=1, warmup_steps=0),
    )
    initial = create_model(config.model, rngs=nnx.Rngs(42))
    save_inference(initial, config, root / "initial", provenance="test fixture")
    (root / "configs").mkdir()
    save_config(config, root / "configs/band.json")
    write(root / "plan.json", dict(initial=str(root / "initial")))
    y = np.asarray(target())
    source = np.ones_like(y)
    source[:, 8:24, 10:20] = -0.2
    return config, prepared_inputs(y, source, np.full_like(y, 0.5), 2)


def test_actual_gap_update_full_state_and_next_update_restore(tmp_path):
    _, prepared = fixture(tmp_path)
    arms = make_arms(tmp_path)
    before = jax.tree.map(np.asarray, nnx.state(arms["band"]["model"], nnx.Param))
    metrics = update(arms["band"], prepared)
    assert metrics["paper_band"] > 0 and metrics["gradient_norm"] > 0
    assert any(
        np.any(a != b)
        for a, b in zip(
            jax.tree.leaves(before),
            jax.tree.leaves(nnx.state(arms["band"]["model"], nnx.Param)),
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
    assert update(arms["band"], prepared) == update(restored["band"], prepared)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 2),
        lineart_state.payload(restored, 2),
    )


def test_generic_trainers_compute_physical_values_for_gap_only_loss(tmp_path):
    from lineartgen.training.lineart_steps import objective
    from lineartgen.training.steps import make_steps

    config, _ = fixture(tmp_path)
    config = config.model_copy(update={"loss": LossConfig(paper_band_weight=0.1)})
    prepared = prepare_batch(config)(target(), jax.random.PRNGKey(4), 2)
    model = create_model(config.model, rngs=nnx.Rngs(42))
    value, parts, _ = objective(model, prepared, None, config)
    assert np.isfinite(float(value)) and "paper_band" in parts
    metrics, grads = make_steps(config)[0](model, prepared)
    assert np.isfinite(float(metrics["loss"])) and "paper_band" in metrics
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(grads))


@pytest.mark.parametrize(
    "changes",
    [
        {"paper_band_weight": -1},
        {"paper_band_weight": float("nan")},
        {"paper_band_radius": 0},
        {"paper_band_radius": 9},
    ],
)
def test_invalid_clearance_config_rejected(changes):
    with pytest.raises(ValueError):
        LossConfig(**changes)
