import jax
import numpy as np
from campaign_protocol import write
from diagnose_learnability import update
from flax import nnx
from lineartgen_runtime.sampling import pyramid
from source_fit_diagnostic import METHODS, fit_checks, forward, make_arms, prepared_inputs

from lineartgen.config import ExperimentConfig, LossConfig, ModelConfig, TrainConfig, save_config
from lineartgen.evaluation_sampling import runtime_inputs
from lineartgen.models import create_model
from lineartgen.models.serialization import save_inference
from lineartgen.training import checkpoints, lineart_state


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
        loss=LossConfig(
            pixel_loss="interval_l1", opacity_deficit_weight=0.3, opacity_deficit_preclip=True
        ),
        train=TrainConfig(steps=10, batch_size=2, accumulate_steps=1, warmup_steps=0),
    )
    initial = create_model(config.model, rngs=nnx.Rngs(42))
    save_inference(initial, config, root / "initial", provenance="test fixture")
    write(root / "plan.json", dict(initial=str(root / "initial")))
    (root / "configs").mkdir()
    save_config(config, root / "configs/small.json")
    wide = config.model_copy(
        update={
            "model": config.model.model_copy(
                update=dict(
                    channels=(4, 8),
                    detail_channels=8,
                    final_decoder_channels=4,
                    head_hidden_channels=6,
                )
            )
        }
    )
    save_config(wide, root / "configs/wide.json")
    clean = np.ones((2, 32, 32, 1), np.float32)
    clean[:, 8:24, 14:17] = -1
    source = clean * 0.4 + 0.6
    strength = np.full_like(clean, 0.5)
    return prepared_inputs(clean, source, strength, 2)


def test_exact_runtime_preparation_and_teacher_separation():
    from source_fit_diagnostic import SEED

    source = np.random.default_rng(13).uniform(-1, 1, (2, 32, 32, 1)).astype(np.float32)
    target = np.ones_like(source)
    strength = np.full_like(source, 0.5)
    for depth in (1, 2, 3):
        prepared = prepared_inputs(target, source, strength, depth)
        noisy, levels = runtime_inputs(source, strength, SEED + depth, depth)
        jax.tree.map(np.testing.assert_array_equal, prepared["noisy"], noisy)
        jax.tree.map(np.testing.assert_array_equal, prepared["levels"], levels)
        jax.tree.map(
            np.testing.assert_array_equal, prepared["guides"], tuple(pyramid(source, depth))
        )
        np.testing.assert_array_equal(prepared["target"], target)
        changed = prepared_inputs(-target, source, strength, depth)
        for key in ("noisy", "levels", "guides"):
            jax.tree.map(np.testing.assert_array_equal, prepared[key], changed[key])


def test_actual_updates_all_bottlenecks_widened_and_next_step_restore(tmp_path):
    prepared = fixture(tmp_path)
    arms = make_arms(tmp_path)
    a, b = [np.asarray(forward(arm["model"], prepared)) for arm in arms.values()]
    np.testing.assert_allclose(a, b, atol=1e-5, rtol=0)
    before = {
        m: jax.tree.map(np.asarray, nnx.state(a["model"], nnx.Param)) for m, a in arms.items()
    }
    for m in METHODS:
        metrics = update(arms[m], prepared)
        assert np.isfinite(metrics["loss"]) and metrics["gradient_norm"] > 0
        assert any(
            np.any(x != y)
            for x, y in zip(
                jax.tree.leaves(before[m]),
                jax.tree.leaves(nnx.state(arms[m]["model"], nnx.Param)),
                strict=True,
            )
        )
    with checkpoints.manager(tmp_path / "checkpoint") as saved:
        lineart_state.save(saved, arms, 1)
        restored = make_arms(tmp_path)
        assert lineart_state.restore(saved, restored) == 1
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 1),
        lineart_state.payload(restored, 1),
    )
    for m in METHODS:
        assert update(arms[m], prepared) == update(restored[m], prepared)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 2),
        lineart_state.payload(restored, 2),
    )


def test_fitting_requires_each_training_case_and_does_not_gate_probe():
    rows = [
        dict(
            method="small",
            split="train",
            case=i,
            ink_f1_alpha50=1.0,
            ink_deficit_fraction=0.0,
            halo_opacity_excess=0.0,
            profile_width_mae_px=0.0,
            connection_cut_rate=0.0,
            connection_bridge_rate=0.0,
        )
        for i in range(4)
    ]
    rows.append(dict(rows[0], case=4, split="probe", ink_f1_alpha50=0.0))
    assert all(c["passed"] for c in fit_checks(rows))
    rows[3]["connection_bridge_rate"] = 0.01
    checks = fit_checks(rows)
    assert len([c for c in checks if not c["passed"]]) == 1
    assert [c for c in checks if not c["passed"]][0]["case"] == 3
    assert not any(c["case"] == 4 for c in checks)
