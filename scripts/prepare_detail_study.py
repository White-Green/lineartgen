"""Prepare paired broader trajectories for the selected width16 detail branch."""

import argparse
import shutil
from pathlib import Path

import jax
import numpy as np
import orbax.checkpoint as ocp
from campaign_protocol import read, sha, write
from parameter_blend_study import original_source
from prepare_source_guide_study import relocate

from lineartgen.config import load_config, save_config
from lineartgen.data.pipeline import dataset_description, discover, split_experiment
from lineartgen.models.serialization import load_inference
from lineartgen.training import lineart_state
from lineartgen.training.lineart_study import create_arm, study_methods


def prepare(coordinator, strokes, mixed, control, fitting):
    if coordinator.exists():
        raise FileExistsError(coordinator)
    for root in (strokes, mixed, control):
        original_source(root)
    fit = read(fitting / "matched-results.json")
    assert next(x for x in fit["results"] if x["method"] == "warm_detail16")["numerical_probe_pass"]
    review = fitting / "evaluation/step-1200/visual-review/review.json"
    assert read(review)["selected_for_broader_study"] == "warm_detail16"
    parents = {"strokes": (strokes, "source_control"), "mixed": (mixed, "mixed_source_control")}
    roots = {f: coordinator.parent / f"detail-{f}-20261008-screen" for f in parents}
    assert all(not p.exists() for p in roots.values())
    plan = dict(
        seed=20261007,
        independent_confirmation_seed=20261005,
        detail_channels=16,
        parameters=130605,
        step=2000,
        total_schedule_steps=10000,
        primary_weights="ema",
        plan_document_sha256=sha(Path("docs/detail-head-broad-plan-20261008.md")),
        local_evidence=dict(
            root=str(fitting),
            results_sha256=sha(fitting / "matched-results.json"),
            review_sha256=sha(review),
            checkpoint_reused=False,
        ),
        trained_endpoints={f: dict(root=str(roots[f]), method=f"detail_{f}") for f in parents},
        parents={
            f: dict(root=str(p), method=n, source_sha256=sha(p / "source.json"))
            for f, (p, n) in parents.items()
        },
        control=dict(
            root=str(control), method="control", source_sha256=sha(control / "source.json")
        ),
        selection=(
            "Inherited EMA 2k screen and actual native review; eligible exact-state "
            "10k continuation, then strict final gates. No blends."
        ),
        confirmation=(
            "Only final passing candidates proceed to seed20261005 and sealed fresh draws."
        ),
        intervention=(
            "Only model.detail_channels=16 against the paired raw-guide control; "
            "original100k plus zero-output additions."
        ),
        acceptance_sha256=sha(control / "acceptance.json"),
    )
    write(coordinator / "plan.json", plan)
    for family, (parent, oldname) in parents.items():
        root = roots[family]
        (root / "configs").mkdir(parents=True)
        for name in (
            "dataset.json",
            "evaluation.json",
            "acceptance.json",
            "supplemental-panels.json",
            "confirmation-plan.json",
        ):
            assert read(parent / name) == read(control / name)
            shutil.copyfile(parent / name, root / name)
        write(root / "plan.json", plan)
        c = relocate(parent / f"configs/{oldname}.json", parent.parent.parent)
        assert c.train.seed == 20261007 and c.train.steps == 10000
        assert c.model.source_conditioning and c.model.guide_encoding == "raw"
        assert c.model.detail_channels == c.model.residual_scale_floor == 0
        assert c.sketch.mode == family and c.loss.contrast_weight == 0.01
        name = f"detail_{family}"
        updated = c.model_copy(update={"model": c.model.model_copy(update={"detail_channels": 16})})
        save_config(updated, root / f"configs/{name}.json")
        save_config(updated, root / "config.json")
        original = read(parent / "protocol.json")
        protocol = {
            k: original[k]
            for k in (
                "initial",
                "initial_files",
                "total_schedule_steps",
                "patch_size",
                "prior_ramp_steps",
                "selected_panels",
            )
        }
        protocol.update(
            methods=[name],
            milestones=[2000],
            primary_weights="ema",
            step=2000,
            plan_sha256=sha(root / "plan.json"),
            counterpart=dict(root=str(parent), method=oldname),
            next_phase="Fixed 2k screen; only eligible exact-state continuation to10k",
        )
        write(root / "protocol.json", protocol)
        write(root / "calibration.json", dict(coefficients={}, no_new_coefficient_fitting=True))
        replay = root / "baseline-replay"
        (replay / "configs").mkdir(parents=True)
        save_config(c, replay / f"configs/{oldname}.json")
        save_config(c, replay / "config.json")
        write(replay / "protocol.json", original | dict(methods=[oldname]))
        write(replay / "calibration.json", dict(coefficients={}))
        sources = discover(c.data)
        train, dev, final = split_experiment(sources, c.data, c.train.seed)
        assert dataset_description(sources, train, dev, final) == read(root / "dataset.json")
        shutil.copytree(parent / "evaluation/step-0-ema", root / "evaluation/step-0-ema")
    print(
        dict(coordinator=str(coordinator), roots={k: str(v) for k, v in roots.items()}), flush=True
    )


def audit_replay(root):
    protocol = read(root / "protocol.json")
    parent, method = Path(protocol["counterpart"]["root"]), protocol["counterpart"]["method"]
    initial, _ = load_inference(protocol["initial"])

    def restore(directory):
        names = study_methods(read(directory / "protocol.json"))
        arms = {n: create_arm(load_config(directory / f"configs/{n}.json"), initial) for n in names}
        with ocp.CheckpointManager(
            (directory / "preflight/checkpoints").resolve(),
            options=ocp.CheckpointManagerOptions(read_only=True),
        ) as saved:
            assert lineart_state.restore(saved, arms, 6) == 6
        return {method: arms[method]}

    before, after = restore(parent), restore(root / "baseline-replay")
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(before, 6),
        lineart_state.payload(after, 6),
    )
    initial_audit = read(root / "initial-guide-audit.json")
    assert initial_audit["all_exact"] and len(initial_audit["rows"]) == 32
    write(
        root / "baseline-replay-audit.json",
        dict(
            full_model_optimizer_ema_exact=True,
            step=6,
            method=method,
            parent=str(parent),
            old_preflight_sha256=sha(parent / "preflight/results.json"),
            new_preflight_sha256=sha(root / "baseline-replay/preflight/results.json"),
            initial_forward_audit_sha256=sha(root / "initial-guide-audit.json"),
        ),
    )
    print(dict(root=str(root), replay_exact=True, initial_forward_exact=True), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "audit-replay"])
    p.add_argument("--root", type=Path, required=True)
    for n in ("strokes", "mixed", "control", "fitting"):
        p.add_argument("--" + n, type=Path)
    a = p.parse_args()
    if a.action == "prepare":
        prepare(
            a.root.resolve(),
            a.strokes.resolve(),
            a.mixed.resolve(),
            a.control.resolve(),
            a.fitting.resolve(),
        )
    else:
        audit_replay(a.root.resolve())
