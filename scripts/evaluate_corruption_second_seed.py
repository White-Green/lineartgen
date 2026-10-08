"""Construct the preselected blend under seed20261006 and compare its matched control."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
from assess_ink_structure_study import assess
from campaign_protocol import read, sha, write
from corruption_blend_study import validate
from evaluate_lineart_loss_study import evaluate
from flax import nnx
from parameter_blend_study import freeze, original_source, verify_endpoints
from prepare_corruption_second_seed import historical_source
from summarize_lineart_loss_study import audit, differences, summarize

from lineartgen.config import load_config, save_config
from lineartgen.models.blending import blend_parameters, compatible_config
from lineartgen.models.serialization import load_inference, save_inference


def prepare(root, coordinator):
    plan = read(coordinator / "plan.json")
    assert plan["alpha"] == 1 / 16 and plan["seed"] == 20261006
    assert read(coordinator / "workflow-status.json")["stage"] == (
        "endpoint_training_complete_blend_evaluation_pending"
    )
    assert read(coordinator / "baseline-replay-audit.json")["full_model_optimizer_ema_exact"]
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(root)
    baseline = Path(plan["baseline_root"])
    historical_source(baseline)
    endpoints = {}
    checks = []
    for tag, declaration in plan["endpoint_roots"].items():
        parent, method = Path(declaration["root"]), declaration["method"]
        original_source(parent)
        protocol = read(parent / "protocol.json")
        assert protocol["independent_fine_tuning_seed"] == 20261006
        checks.append(read(parent / "checkpoint.json"))
        audited = read(parent / "checkpoint-audits/step-2000.json")
        assert audited["snapshots_checked"] and audited["step"] == 2000
        assert len(audited["records"]) == 1
        record = audited["records"][0]
        assert record["method"] == method and record["all_states_finite"]
        endpoints[tag] = dict(
            root=str(parent), method=method, source_sha256=sha(parent / "source.json"), modes={}
        )
        for mode in ("ema", "raw"):
            folder = parent / f"snapshots/step-2000/{method}/{mode}"
            files = {n: sha(folder / n) for n in ("weights.npz", "metadata.json", "config.json")}
            assert files["weights.npz"] == record["snapshots"][mode]["weights_sha256"]
            assert record["snapshots"][mode]["checkpoint_exact"]
            endpoints[tag]["modes"][mode] = dict(path=str(folder), files=files)
    assert all(c["step"] == 2000 for c in checks)
    assert len({c["clean_sha256"] for c in checks}) == 1
    (root / "configs").mkdir(parents=True)
    for name in (
        "plan.json",
        "dataset.json",
        "evaluation.json",
        "acceptance.json",
        "supplemental-panels.json",
        "confirmation-plan.json",
        "selection-review.json",
        "baseline-replay-audit.json",
    ):
        shutil.copyfile(coordinator / name, root / name)
    config = load_config(Path(endpoints["strokes"]["root"]) / "config.json")
    save_config(config, root / "config.json")
    write(
        root / "endpoint-seal.json",
        dict(
            step=2000,
            endpoints=endpoints,
            plan_sha256=sha(root / "plan.json"),
            same_training_final_batch_sha256=checks[0]["clean_sha256"],
        ),
    )
    records = []
    for mode in ("ema", "raw"):
        first, a = load_inference(endpoints["strokes"]["modes"][mode]["path"])
        second, b = load_inference(endpoints["mixed"]["modes"][mode]["path"])
        compatible_config(a.model, b.model)
        assert a.train.seed == b.train.seed == 20261006
        model = blend_parameters(first, second, 1 / 16)
        folder = root / f"snapshots/step-2000/candidate/{mode}"
        save_inference(
            model,
            config,
            folder,
            provenance=(
                f"Independent fine-tuning seed20261006, 2k, fixed alpha=1/16 corruption blend, "
                f"{mode}; no blended optimizer state"
            ),
        )
        loaded, _ = load_inference(folder)
        jax.tree.map(np.testing.assert_array_equal, nnx.state(model), nnx.state(loaded))
        files = {n: sha(folder / n) for n in ("weights.npz", "metadata.json", "config.json")}
        records.append(
            dict(method="candidate", mode=mode, alpha=1 / 16, roundtrip_exact=True, files=files)
        )
        old = plan["baseline_modes"][mode]
        source = Path(old["path"])
        assert all(sha(source / n) == digest for n, digest in old["files"].items())
        destination = root / f"snapshots/step-2000/control/{mode}"
        shutil.copytree(source, destination)
        records.append(
            dict(method="control", mode=mode, original_files_copied_exact=True, files=old["files"])
        )
    save_config(config, root / "configs/candidate.json")
    control = load_config(root / "snapshots/step-2000/control/ema/config.json")
    save_config(control, root / "configs/control.json")
    source_protocol = read(Path(endpoints["strokes"]["root"]) / "protocol.json")
    protocol = {
        k: source_protocol[k]
        for k in (
            "initial",
            "initial_files",
            "selected_panels",
            "total_schedule_steps",
            "patch_size",
        )
    }
    protocol.update(
        methods=["control", "candidate"],
        milestones=[2000],
        step=2000,
        alpha_map={"candidate": 1 / 16},
        seed=20261006,
        plan_sha256=sha(root / "plan.json"),
        endpoint_seal_sha256=sha(root / "endpoint-seal.json"),
        study_type="Independent fine-tuning seed confirmation of fixed blend; strict final gates",
    )
    write(root / "protocol.json", protocol)
    write(
        root / "construction.json",
        dict(
            records=records,
            optimizer_states_created=False,
            script_sha256=sha(Path(__file__)),
            blend_source_sha256=sha(Path("src/lineartgen/models/blending.py")),
        ),
    )
    shutil.copytree(baseline / "evaluation/step-0-ema", root / "evaluation/step-0-ema")
    freeze(root)
    print(dict(root=str(root), alpha=1 / 16, seed=20261006), flush=True)


def compare(root):
    protocol, _ = verify_endpoints(root)
    assert protocol["alpha_map"] == {"candidate": 1 / 16}
    plan, reference = read(root / "plan.json"), read(root / "evaluation.json")
    baseline = Path(plan["baseline_root"])
    historical_source(baseline)
    manifest = read(Path(reference["root"]) / "cases.json")
    assert sha(Path(reference["root"]) / "cases.json") == reference["cases_sha256"]
    initial, artifacts = audit(root, 0, "ema", reference, len(manifest["cases"]))
    milestones, gates = {}, {}
    for mode in ("ema", "raw"):
        result, files = audit(root, 2000, mode, reference, len(manifest["cases"]))
        artifacts.update(files)
        old, files = audit(baseline, 2000, mode, reference, len(manifest["cases"]))
        artifacts.update(files)
        control = [r for r in result["results"] if r["method"] == "control"]
        assert control == [r for r in old["results"] if r["method"] == "control"], (
            "Historical control inference or metrics differ"
        )
        rows = result["results"]
        summary = summarize(rows)
        milestones[f"2000-{mode}"] = dict(
            summary=summary,
            paired=differences(rows, rows, manifest, "control")
            + differences(rows, initial["results"], manifest, "initial"),
        )
        gates[mode] = assess(summary, read(root / "acceptance.json"), "candidate", screening=False)
    analysis = root / "analysis-with-baselines-2000.json"
    write(
        analysis,
        dict(
            through=2000,
            seed=20261006,
            alpha=1 / 16,
            milestones=milestones,
            artifacts=artifacts,
            historical_control_all_1562_records_exact_both_modes=True,
            analysis_source_sha256=sha(Path(__file__)),
            limitations=(
                "Independent fine-tuning seed, shared original 100k initialization "
                "and development cases; fresh confirmation still pending"
            ),
        ),
    )
    write(
        root / "quality-gate-with-baselines-2000.json",
        dict(
            step=2000,
            seed=20261006,
            results=[gates["ema"]],
            diagnostics=gates,
            ranked_provisional_candidates=["candidate"]
            if gates["ema"]["numerical_gate_passed"]
            else [],
            eligibility="strict final",
            analysis_sha256=sha(analysis),
            rules_sha256=sha(root / "acceptance.json"),
        ),
    )
    print(
        {
            mode: dict(
                passed=x["numerical_gate_passed"],
                failures=len(x["failures"]),
                halo=x["primary_halo_gains"],
            )
            for mode, x in gates.items()
        },
        flush=True,
    )


def run(root):
    verify_endpoints(root)
    validate(root)
    for mode in ("ema", "raw"):
        evaluate(root, 2000, mode)
    evaluate(root, 2000, "ema", diagnostic=True)
    for mode in ("ema", "raw"):
        for script in ("evaluate_lineart_blank_study", "render_lineart_loss_supplement"):
            subprocess.run(
                [
                    sys.executable,
                    f"scripts/{script}.py",
                    "--root",
                    str(root),
                    "--step",
                    "2000",
                    "--mode",
                    mode,
                ],
                check=True,
            )
    compare(root)
    write(root / "workflow-status.json", dict(stage="numerical_complete_visual_review_pending"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "run"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--coordinator", type=Path)
    a = p.parse_args()
    if a.action == "prepare":
        prepare(a.root.resolve(), a.coordinator.resolve())
    else:
        run(a.root.resolve())
