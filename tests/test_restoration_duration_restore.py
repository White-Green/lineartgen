import jax
import jax.numpy as jnp
import numpy as np
from campaign_protocol import read, write
from diagnose_learnability import update
from flax import nnx
from train_procedural_structure import prepare_batch
from train_restoration_duration import METHODS, configure_arm, make_arms, restore_parents

from lineartgen.config import ExperimentConfig, save_config
from lineartgen.models import create_model
from lineartgen.models.serialization import save_inference
from lineartgen.training import checkpoints, lineart_state


def test_full_parent_restore_and_next_update_preserve_optimizer_and_data_clock(tmp_path):
    config = ExperimentConfig.model_validate(
        dict(
            model=dict(
                channels=(2, 4),
                convolutions_per_block=2,
                output_mode="residual",
                source_conditioning=True,
                residual_base="source",
                feature_input="source",
                sampling_mode="one_pass",
                detail_channels=4,
            ),
            diffusion=dict(scale_loss_multiplier=0, pattern_weights=(1, 1, 1, 0)),
            matching=dict(method="independent"),
            loss=dict(
                pixel_loss="interval_l1", opacity_deficit_weight=0.3, opacity_deficit_preclip=True
            ),
            sketch=dict(mode="strokes", probability=1),
            train=dict(steps=10, batch_size=2, accumulate_steps=1, warmup_steps=0),
        )
    )
    initial = create_model(config.model, rngs=nnx.Rngs(42))
    save_inference(initial, config, tmp_path / "initial", provenance="test fixture")
    root = tmp_path / "extension"
    (root / "configs").mkdir(parents=True)
    old_arms = {}
    parents = {}
    clean = jnp.ones((2, 16, 16, 1)).at[:, 3:14, 7:9].set(-1)
    for name, blend in zip(METHODS, (1.0, 0.5), strict=True):
        old = config.model_copy(
            update={"sketch": config.sketch.model_copy(update={"dilation_blend": blend})}
        )
        parent = tmp_path / name
        (parent / "configs").mkdir(parents=True)
        save_config(old, parent / f"configs/{name}.json")
        arm = configure_arm(old, initial)
        old_arms[name] = arm
        preparer = prepare_batch(old)
        for step in range(10):
            update(arm, preparer(clean, jax.random.fold_in(jax.random.PRNGKey(91), step), 2))
        with checkpoints.manager(parent / "milestone-checkpoints") as saved:
            lineart_state.save(saved, {name: arm}, 10)
        new = old.model_copy(
            update={
                "train": old.train.model_copy(update={"steps": 20, "learning_rate_decay_steps": 10})
            }
        )
        save_config(new, root / f"configs/{name}.json")
        parents[name] = dict(root=str(parent), method=name, step=10)
    write(root / "plan.json", dict(initial=str(tmp_path / "initial"), parents=parents, start=10))
    restored = make_arms(root)
    restore_parents(root, restored)
    assert read(root / "parent-state-audit.json")["model_optimizer_ema_exact"]
    for name in METHODS:
        key = jax.random.fold_in(jax.random.PRNGKey(91), 10)
        old_batch = prepare_batch(old_arms[name]["config"])(clean, key, 2)
        new_batch = prepare_batch(restored[name]["config"])(clean, key, 2)
        jax.tree.map(np.testing.assert_array_equal, old_batch, new_batch)
        old_metrics = update(old_arms[name], old_batch)
        new_metrics = update(restored[name], new_batch)
        assert old_metrics == new_metrics
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(old_arms, 11),
        lineart_state.payload(restored, 11),
    )
    with checkpoints.manager(root / "checkpoints") as saved:
        lineart_state.save(saved, restored, 11)
        again = make_arms(root)
        assert lineart_state.restore(saved, again, 11) == 11
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(restored, 11),
        lineart_state.payload(again, 11),
    )
