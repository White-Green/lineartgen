"""Continue the fixed fresh-noise diagnostic without resetting training state."""

import argparse
import datetime
import fcntl
import shutil
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from diagnose_learnability import arms_from_root, snapshot
from fit_fresh_noise import render_transfer, train, transfer, verify_fresh
from fit_single_cases import cached_inputs, evaluate, render

from lineartgen.training import checkpoints, lineart_state


def prepare(root, resume):
    plan = verify_fresh(resume)
    assert read(resume / "workflow-status.json")["stage"] == "complete_diagnostic_insufficient"
    assert not root.exists()
    (root / "configs").mkdir(parents=True)
    names = [
        "cached-inputs.npz",
        "train.json",
        "probe.json",
        "input-audit.json",
        "cached-replay-audit.json",
        "initial-forward-audit.json",
    ]
    names += [f"configs/{name}.json" for name in plan["cases"]]
    for name in names:
        shutil.copyfile(resume / name, root / name)
    files = names + [
        "source.json",
        "plan.json",
        "training/restore-audit-step-800.json",
        "evaluation/step-800/predictions.npz",
        "evaluation/step-800/metrics.json",
        "transfer/step-800/predictions.npz",
        "transfer/step-800/metrics.json",
        "visual-review/review.json",
        "transfer/visual-review/review.json",
    ]
    checkpoint = resume / "training/checkpoints/800"
    assert checkpoint.is_dir()
    files += [str(p.relative_to(resume)) for p in checkpoint.rglob("*") if p.is_file()]
    for destination in ("training", "preflight"):
        shutil.copytree(checkpoint, root / destination / "checkpoints/800")
    plan.update(
        resume_from=str(resume),
        resume_step=800,
        final_step=4000,
        milestones=[800, 1600, 4000],
        schedule_horizon=800,
        resume_files={n: sha(resume / n) for n in files},
        plan_document_sha256=sha(Path("docs/noise-convergence-plan-20261009.md")),
        primary=(
            "raw one-step depth3 at4000 on the same eight diagnostic noise draws "
            "per seen TRAIN case"
        ),
    )
    write(root / "plan.json", plan)
    print(dict(prepared=str(root), resume_step=800, final_step=4000), flush=True)


def restore_from(root, relative):
    arms = arms_from_root(root)
    with checkpoints.manager(root / relative, keep=3) as saved:
        last = lineart_state.restore(saved, arms)
    return arms, last


def initial_audit(root):
    plan = verify_fresh(root)
    resume = Path(plan["resume_from"])
    original, old_step = restore_from(resume, "training/checkpoints")
    assert old_step == 800
    schedules = {}
    for destination in ("training", "preflight"):
        actual, step = restore_from(root, destination + "/checkpoints")
        assert step == 800
        jax.tree.map(
            np.testing.assert_array_equal,
            lineart_state.payload(actual, step),
            lineart_state.payload(original, old_step),
        )
        for name, arm in actual.items():
            values = [float(arm["schedule"](s)) for s in (800, 1600, 4000)]
            np.testing.assert_allclose(values, 1e-5, rtol=1e-6, atol=0)
            assert arm["config"].train.steps == 800
            schedules[name] = values
    snapshot(root, actual, 800)
    evaluate(root, actual, cached_inputs(root), 800)
    a = np.load(root / "evaluation/step-800/predictions.npz")
    b = np.load(resume / "evaluation/step-800/predictions.npz")
    assert set(a.files) == set(b.files)
    for name in a.files:
        np.testing.assert_array_equal(a[name], b[name])
    write(
        root / "resume-audit.json",
        dict(
            all_state_exact=True,
            cached_predictions_exact=len(a.files),
            step=800,
            learning_rates=schedules,
        ),
    )
    print("resume state and all24 cached predictions exact; LR stays1e-5", flush=True)


def compare_six(root):
    actual, step = restore_from(root, "training/checkpoints")
    preflight, other = restore_from(root, "preflight/checkpoints")
    assert step == other == 806
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(actual, step),
        lineart_state.payload(preflight, other),
    )
    write(root / "resumed-six-update-audit.json", dict(all_state_exact=True, start=800, end=806))
    print("actual resumed six updates exactly match preflight", flush=True)


def run(root):
    def status(stage):
        write(
            root / "workflow-status.json",
            dict(
                stage=stage,
                updated_utc=datetime.datetime.now(datetime.UTC).isoformat(),
                adoption=False,
            ),
        )
        print(stage, flush=True)

    with (root / "coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            status("resume_audit")
            initial_audit(root)
            status("resumed_preflight")
            train(root, preflight=True)
            for end in (801, 806, 1600, 4000):
                status(f"training_to_{end}")
                train(root, end=end)
                if end == 806:
                    compare_six(root)
                if end in (1600, 4000):
                    status(f"transfer_{end}")
                    transfer(root, end)
            status("rendering")
            render(root, milestones=(800, 1600, 4000), final_step=4000)
            render_transfer(root, step=4000)
            verify_fresh(root)
            status("numerical_complete_native_review_pending")
        except BaseException as exc:
            status(f"failed_{type(exc).__name__}")
            raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "run"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    a = p.parse_args()
    if a.action == "prepare":
        prepare(a.root.resolve(), a.resume.resolve())
    else:
        run(a.root.resolve())
