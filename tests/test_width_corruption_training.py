import jax
import jax.numpy as jnp
import numpy as np
from campaign_protocol import write
from diagnose_learnability import update
from flax import nnx
from train_procedural_structure import prepare_batch
from train_width_corruption_interaction import make_arms

from lineartgen.config import (
    DiffusionConfig,
    ExperimentConfig,
    LossConfig,
    ModelConfig,
    NoiseMatchingConfig,
    SketchAugmentationConfig,
    TrainConfig,
    save_config,
)
from lineartgen.models import create_model
from lineartgen.models.serialization import save_inference
from lineartgen.training import checkpoints, lineart_state


def test_width_corruption_study_performs_real_update_and_restores_full_state(tmp_path):
    config = ExperimentConfig(
        model=ModelConfig(
            channels=(2, 4),
            convolutions_per_block=2,
            output_mode="residual",
            source_conditioning=True,
            residual_base="source",
            feature_input="source",
            sampling_mode="one_pass",
            detail_channels=4,
        ),
        diffusion=DiffusionConfig(scale_loss_multiplier=0, pattern_weights=(1, 1, 1, 0)),
        matching=NoiseMatchingConfig(method="independent"),
        loss=LossConfig(
            pixel_loss="interval_l1", opacity_deficit_weight=0.3, opacity_deficit_preclip=True
        ),
        sketch=SketchAugmentationConfig(mode="strokes", probability=1, strong_probability=0.5),
        train=TrainConfig(steps=10, batch_size=2, accumulate_steps=1, warmup_steps=0),
    )
    initial = create_model(config.model, rngs=nnx.Rngs(42))
    save_inference(initial, config, tmp_path / "initial", provenance="test fixture")
    (tmp_path / "configs").mkdir()
    for name, floor in [("combined", 0.05)]:
        save_config(
            config.model_copy(
                update={"loss": config.loss.model_copy(update={"noise_weight_floor": floor})}
            ),
            tmp_path / f"configs/{name}.json",
        )
    write(
        tmp_path / "plan.json",
        dict(
            initial=str(tmp_path / "initial"),
            arms=["combined"],
            warm_start_methods=["combined"],
        ),
    )
    arms = make_arms(tmp_path)
    before = jax.tree.map(np.asarray, nnx.state(arms["combined"]["model"], nnx.Param))
    clean = jnp.ones((2, 32, 32, 1)).at[:, 8:25, 14:17].set(-1)
    for arm in arms.values():
        prepared = prepare_batch(arm["config"])(clean, jax.random.PRNGKey(3), 2)
        metrics = update(arm, prepared)
        assert np.isfinite(metrics["loss"]) and metrics["gradient_norm"] > 0
    after = nnx.state(arms["combined"]["model"], nnx.Param)
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
