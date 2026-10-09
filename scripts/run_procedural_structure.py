"""Sequential coordinator; does not approve a model or substitute for image review."""

import argparse
import fcntl
import subprocess
import sys
import time
from pathlib import Path

from campaign_protocol import read, write


def run(root, previous):
    with (root / "coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(root / "workflow-status.json", dict(stage="waiting_for_previous_gpu_run"))
        while True:
            status = previous / "workflow-status.json"
            if status.exists() and read(status)["stage"] in (
                "trained_pending_visual_review",
                "complete_diagnostic_insufficient",
                "complete_rejected",
            ):
                break
            time.sleep(10)
        actions = [
            ("preflight", ["scripts/train_procedural_structure.py", "preflight"]),
            ("training1000", ["scripts/train_procedural_structure.py", "train", "--end", "1000"]),
            ("broad1000", ["scripts/evaluate_procedural_structure.py", "--step", "1000"]),
            ("training4000", ["scripts/train_procedural_structure.py", "train", "--end", "4000"]),
            ("broad4000", ["scripts/evaluate_procedural_structure.py", "--step", "4000"]),
        ]
        for stage, args in actions:
            write(root / "workflow-status.json", dict(stage=stage))
            with (root / f"{stage}.log").open("a") as log:
                result = subprocess.run(
                    [sys.executable, *args, "--root", str(root)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            if result.returncode:
                write(
                    root / "workflow-status.json",
                    dict(stage="error", action=stage, returncode=result.returncode),
                )
                raise RuntimeError(f"{stage} failed; see {root / (stage + '.log')}")
        write(
            root / "workflow-status.json",
            dict(stage="numerical_complete_pending_review", step=4000),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.previous.resolve())
