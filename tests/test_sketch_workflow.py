import copy
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import jax
import numpy as np
import orbax.checkpoint as ocp
import pytest
from flax import nnx
from PIL import Image

from lineartgen.config import ExperimentConfig, save_config
from lineartgen.models import create_model
from lineartgen.models.serialization import load_inference, save_inference
from lineartgen.training import checkpoints


def script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sketch_gate_checks_complete_grid_and_protected_input():
    workflow = script("run_sketch_experiment")
    report = dict(
        checkpoint_step=2000,
        cases=1562,
        results=[
            dict(
                case=c, method=m, subnet=s, denoise_steps=n, protected_max_error=0.0, active_mae=0.1
            )
            for c, m, s, n in itertools.product(range(1562), workflow.METHODS, range(3), (1, 5))
        ],
        summary=[
            dict(subset=k, method=m, subnet=s, denoise_steps=1, ink_f1=0.7)
            for k, m, s in itertools.product(
                ("matched_real", "geometry"), workflow.METHODS, range(3)
            )
        ],
    )
    assert workflow.pilot_gate([report, report])["passed"]
    for issue in ("missing", "duplicate", "nonfinite", "protected", "collapse"):
        bad = copy.deepcopy(report)
        if issue == "missing":
            bad["results"].pop()
        elif issue == "duplicate":
            bad["results"][-1] = bad["results"][0]
        elif issue == "nonfinite":
            bad["results"][0]["active_mae"] = float("nan")
        elif issue == "protected":
            bad["results"][0]["protected_max_error"] = 0.01
        else:
            bad["summary"][0]["ink_f1"] = 0.1
        assert not workflow.pilot_gate([bad, report])["passed"]


@pytest.mark.parametrize(
    "study",
    ["sketch", "sketch_rates", "geometry_data", "boundary_loss", "boundary_half", "cldice_loss"],
)
def test_paired_warmstart_resumes_model_optimizer_and_ema_exactly(tmp_path, monkeypatch, study):
    dataset = tmp_path / "dataset" / "images"
    dataset.mkdir(parents=True)
    for index in range(5):
        gray = np.full((20, 20), 255, np.uint8)
        gray[:, 7 + index : 9 + index] = 0
        Image.fromarray(gray).save(dataset / f"{index}.png")
    config = ExperimentConfig.model_validate(
        {
            "model": {
                "architecture": "nested",
                "channels": [2],
                "nested_channels": [[3], [4]],
                "convolutions_per_block": 1,
                "output_mode": "residual",
            },
            "data": {
                "root": dataset.parent,
                "scale_dirs": ["images"],
                "crop_size": 16,
                "validation_count": 1,
                "workers": 1,
            },
            "diffusion": {"pattern_weights": [1, 1, 1, 0], "sample_steps": 1},
            "matching": {"method": "independent", "pool_size": 2},
            "loss": {"noise_weight_floor": 0.05, "contrast_active_only": True},
            "train": {
                "steps": 3,
                "warmup_steps": 0,
                "batch_size": 1,
                "accumulate_steps": 2,
                "evaluate_every": 3,
                "checkpoint_every": 1,
                "validation_batches": 1,
            },
        }
    )
    config_path = tmp_path / "config.json"
    if study in ("geometry_data", "boundary_loss", "boundary_half", "cldice_loss"):
        config = config.model_copy(
            update={
                "sketch": config.sketch.model_copy(update={"mode": "strokes", "probability": 0.5})
            }
        )
    if study in ("boundary_loss", "boundary_half", "cldice_loss"):
        config = config.model_copy(
            update={
                "geometry": config.geometry.model_copy(update={"probability": 0.2}),
                "loss": config.loss.model_copy(
                    update=(
                        {"cldice_weight": 0.2, "cldice_iterations": 3}
                        if study == "cldice_loss"
                        else {"boundary_weight": 0.2}
                    )
                ),
            }
        )
    save_config(config, config_path)
    model = create_model(config.model, rngs=nnx.Rngs(9))
    initial = tmp_path / "warmstart"
    save_inference(model, config, initial, provenance="non-default initialization")
    compare = script("compare_nested_training")
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    for root, count, resume in ((full, 3, False), (resumed, 1, False), (resumed, 2, True)):
        flags = ["--resume"] if resume else ["--initialize-from", str(initial)]
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "compare_nested_training",
                "--study",
                study,
                "--config",
                str(config_path),
                "--run-dir",
                str(root),
                "--stop-after",
                str(count),
                *flags,
            ],
        )
        compare.main()
    names = {
        "sketch": ("control", "tone", "blur", "strokes"),
        "sketch_rates": ("p75", "p50", "p25"),
        "geometry_data": ("control", "geometry20"),
        "boundary_loss": ("control", "boundary"),
        "boundary_half": ("half",),
        "cldice_loss": ("control", "centerline"),
    }[study]
    for name in names:
        archived, _ = load_inference(full / "initial" / name)
        jax.tree.map(
            np.testing.assert_array_equal,
            nnx.state(archived, nnx.Param),
            nnx.state(model, nnx.Param),
        )
        states = []
        for root in (full, resumed):
            assert json.loads((root / "progress.json").read_text())["step"] == 3
            with checkpoints.manager(root / name / "checkpoints") as saved:
                states.append(saved.restore(3, args=ocp.args.StandardRestore()))
        jax.tree.map(np.testing.assert_array_equal, *states)

    if study in ("boundary_loss", "cldice_loss"):
        paths = [full / name / "checkpoints" for name in names]
        final_states = []
        for path in paths:
            with checkpoints.manager(path) as saved:
                final_states.append(saved.restore(3, args=ocp.args.StandardRestore()))
        assert any(
            not np.array_equal(a, b)
            for a, b in zip(
                jax.tree.leaves(final_states[0]), jax.tree.leaves(final_states[1]), strict=True
            )
        )

    if study == "boundary_half":
        # Compare single-arm execution with the unchanged two-arm runner, including
        # model, optimizer and EMA states, not merely logged losses.
        paired = tmp_path / "paired"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "compare_nested_training",
                "--study",
                "boundary_loss",
                "--config",
                str(config_path),
                "--run-dir",
                str(paired),
                "--stop-after",
                "3",
                "--initialize-from",
                str(initial),
            ],
        )
        compare.main()
        with checkpoints.manager(paired / "boundary/checkpoints") as saved:
            paired_state = saved.restore(3, args=ocp.args.StandardRestore())
        jax.tree.map(np.testing.assert_array_equal, states[0], paired_state)
