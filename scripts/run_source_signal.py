"""Wait for the fixed control, then run the isolated treatment and both evaluations."""

import argparse
import fcntl
import subprocess
import sys
import time
from pathlib import Path

from campaign_protocol import read, write


def run(root, parent):
    with (root / "coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(root / "workflow-status.json", dict(stage="waiting_for_fixed_control"))
        while True:
            status = parent / "workflow-status.json"
            if status.exists() and read(status)["stage"] in (
                "numerical_complete_pending_review",
                "complete_diagnostic_insufficient",
                "complete_rejected",
                "complete_candidate_screen",
            ):
                break
            if status.exists() and read(status)["stage"] == "error":
                raise RuntimeError("Parent run failed; control must be completed first")
            time.sleep(10)
        commands = [
            ("bind", ["scripts/train_source_signal.py", "bind"]),
            ("control_replay", ["scripts/train_source_signal.py", "replay"]),
            ("preflight", ["scripts/train_source_signal.py", "preflight"]),
            ("training1000", ["scripts/train_source_signal.py", "train", "--end", "1000"]),
            ("broad1000", ["scripts/evaluate_procedural_structure.py", "--step", "1000"]),
            ("audit1000", ["scripts/train_source_signal.py", "audit", "--end", "1000"]),
            ("training4000", ["scripts/train_source_signal.py", "train", "--end", "4000"]),
            ("broad4000", ["scripts/evaluate_procedural_structure.py", "--step", "4000"]),
            ("audit4000", ["scripts/train_source_signal.py", "audit", "--end", "4000"]),
        ]
        for stage, args in commands:
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
                raise RuntimeError(f"{stage} failed; inspect its log")
        write(
            root / "workflow-status.json",
            dict(stage="numerical_complete_pending_review", step=4000),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.parent.resolve())
