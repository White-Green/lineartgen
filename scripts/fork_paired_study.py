"""Continue paired-study candidates while reusing the completed external control."""

import argparse
import shutil
from pathlib import Path

from campaign_protocol import read, sha, write
from fork_ink_structure_study import transfer
from train_lineart_loss_study import snapshot, verify

from lineartgen.config import load_config
from lineartgen.models.serialization import load_inference
from lineartgen.training.lineart_study import create_arm, study_methods


def fork(parent, root, review):
    protocol = verify(parent)
    assert read(parent / "workflow-status.json")["stage"] == (
        "numerical_complete_visual_review_and_report_pending"
    )
    assert read(parent / "analysis-status.json")["stage"] == (
        "numerical_diagnostics_complete_visual_review_pending"
    )
    step = 2000
    gate_path = parent / f"quality-gate-with-baselines-{step}.json"
    gate, decision = read(gate_path), read(review)
    assert decision["quality_gate_sha256"] == sha(gate_path)
    assert decision["inspection_notes"] and decision["inspected_panels"]
    for path, digest in decision["inspected_panels"].items():
        assert sha(path) == digest
    candidates = decision["approved_methods"]
    assert 1 <= len(candidates) <= 2 and len(set(candidates)) == len(candidates)
    assert set(candidates) <= set(gate["ranked_provisional_candidates"])
    assert not set(candidates) & {"control", "no_contrast"}
    methods = candidates
    assert "control" not in study_methods(protocol)
    assert "10000" in protocol["external_baselines"]
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(root)
    root.mkdir(parents=True, exist_ok=True)
    for name in (
        "config.json",
        "dataset.json",
        "evaluation.json",
        "calibration.json",
        "acceptance.json",
        "supplemental-panels.json",
        "confirmation-plan.json",
        "plan.json",
        "baseline-replay-audit.json",
        "supplemental-validation.json",
        "baseline-replay/preflight/results.json",
    ):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(parent / name, root / name)
    shutil.copyfile(review, root / "selection-review.json")
    (root / "configs").mkdir()
    for name in methods:
        shutil.copyfile(parent / "configs" / f"{name}.json", root / "configs" / f"{name}.json")
    protocol = protocol | dict(
        methods=methods,
        milestones=[10000],
        optimizer="Full parent 2k optimizer/model/EMA continuation; original cosine clock 10k",
        next_phase="Review at 10k against external controls and fixed gates before confirmations",
        continuation=dict(
            parent=str(parent),
            step=step,
            parent_source_sha256=sha(parent / "source.json"),
            selection_review_sha256=sha(review),
        ),
    )
    write(root / "protocol.json", protocol)
    initial, _ = load_inference(protocol["initial"])

    def create():
        return {
            name: create_arm(
                load_config(parent / "configs" / f"{name}.json"),
                initial,
                adversarial=name == "patch_prior",
                patch_size=protocol["patch_size"],
            )
            for name in study_methods(read(parent / "protocol.json"))
        }

    restored = transfer(parent / "checkpoints", root / "checkpoints", create, methods, step)
    snapshot(root, restored, step)
    write(
        root / "continuation-audit.json",
        dict(
            parent=str(parent),
            step=step,
            methods=methods,
            model_optimizer_ema_exact=True,
            optimizer_counters_exact=True,
            script_sha256=sha(Path(__file__)),
            parent_checkpoint_files={
                str(p.relative_to(parent)): sha(p)
                for p in sorted((parent / "checkpoints" / str(step)).rglob("*"))
                if p.is_file()
            },
        ),
    )
    # The numerical config is identical. Reuse its actual six-depth GPU preflight,
    # with explicit provenance; the separate audit above checks the actual 2k state.
    preflight = read(parent / "preflight/results.json")
    preflight["updates"] = [r for r in preflight["updates"] if r["method"] in methods]
    preflight["inherited_validation"] = dict(
        parent=str(parent),
        sha256=sha(parent / "preflight/results.json"),
        reason="Same numerical configuration and code; selected methods only",
    )
    write(root / "preflight/results.json", preflight)
    print(dict(root=str(root), step=step, methods=methods, full_state_exact=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    fork(args.parent.resolve(), args.root.resolve(), args.review.resolve())
