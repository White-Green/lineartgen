"""Resume the fixed loss study through 2k/10k and evaluate every completed milestone."""

import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

from campaign_protocol import read, write
from train_lineart_loss_study import verify


def collapse_failures(root, step):
    initial = read(root / "evaluation/step-0-ema/metrics.json")
    current = read(root / f"evaluation/step-{step}-ema/metrics.json")
    reference = {r["case"]: r for r in initial["results"]}
    failures = []
    for method in sorted({r["method"] for r in current["results"]}):
        rows = [r for r in current["results"] if r["method"] == method and r["family"] == "clean"]
        before = sum(reference[r["case"]]["active_ink_f1"] for r in rows) / len(rows)
        after = sum(r["active_ink_f1"] for r in rows) / len(rows)
        if after < min(0.3, before * 0.5):
            failures.append(dict(method=method, initial=before, current=after))
    return failures


def run(root):
    protocol = verify(root)
    assert read(root / "preflight/results.json")["checkpoint_all_states_exact"]

    def command(script, arguments):
        label = script + "-" + "-".join(str(x).replace("/", "_") for x in arguments)
        write(
            root / "workflow-status.json",
            dict(
                stage=label, pid=os.getpid(), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            ),
        )
        with (root / f"{label}.log").open("a") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    f"scripts/{script}.py",
                    "--root",
                    str(root),
                    *map(str, arguments),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    command("evaluate_lineart_loss_study", ["--step", 0])
    for step in protocol["milestones"]:
        if (root / "stop").exists():
            write(root / "workflow-status.json", dict(stage="stopped_by_stop_file"))
            return
        if not all(
            (root / f"snapshots/step-{step}" / name / "ema/weights.npz").is_file()
            for name in ("control", "no_contrast", "regions", "profiles", "patch_prior")
        ):
            command("train_lineart_loss_study", ["--end", step])
            if (root / "stop").exists():
                return
        for mode in ("ema", "raw"):
            command("evaluate_lineart_loss_study", ["--step", step, "--mode", mode])
        failures = collapse_failures(root, step)
        write(
            root / f"gate-{step}.json",
            dict(
                failures=failures,
                passed=not failures,
                meaning="Finite/protected and gross collapse check; not quality acceptance",
            ),
        )
        if failures:
            raise RuntimeError(f"Study requires review after collapse: {failures}")
    command("evaluate_lineart_loss_study", ["--step", protocol["milestones"][-1], "--diagnostic"])
    write(
        root / "workflow-status.json",
        dict(stage="numerical_complete_visual_review_and_report_pending"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    with (root / "controller.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(root)
        except Exception as error:
            write(root / "workflow-status.json", dict(stage="failed", error=str(error)))
            raise


if __name__ == "__main__":
    main()
