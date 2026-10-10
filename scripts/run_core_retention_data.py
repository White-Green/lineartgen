"""Run the frozen 4k core-retention-data experiment, retaining the six-step startup state."""

import fcntl
import subprocess
import sys
from pathlib import Path

from campaign_protocol import read, write
from train_core_retention_data import check

root = Path("runs/core-retention-data-20261010").resolve()
with (root / "coordinator.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = check(root)
    gate = Path(plan["interval_gate"])
    assert read(gate / "final-audit.json")["stage"] == "complete_diagnostic_insufficient"
    assert read(root / "data-audit.json")["crop_resume_exact"]
    assert read(root / "test-results.json")["passed"]
    commands = [
        ("startup6", ["scripts/train_core_retention_data.py", "train", "--end", "6"]),
    ]
    for step in (1000, 4000):
        commands.extend(
            [
                (
                    f"train{step}",
                    ["scripts/train_core_retention_data.py", "train", "--end", str(step)],
                ),
                (f"local{step}", ["scripts/evaluate_procedural_structure.py", "--step", str(step)]),
                (
                    f"control{step}",
                    ["scripts/train_core_retention_data.py", "audit-broad", "--end", str(step)],
                ),
            ]
        )
    commands.extend(
        [
            ("development", ["scripts/evaluate_core_retention_data.py", "evaluate"]),
            ("audit", ["scripts/train_core_retention_data.py", "audit"]),
        ]
    )
    for stage, args in commands:
        write(root / "workflow-status.json", dict(stage=stage))
        with (root / f"{stage}.log").open("a") as log:
            code = subprocess.run(
                [sys.executable, *args, "--root", str(root)], stdout=log, stderr=subprocess.STDOUT
            ).returncode
        if code:
            write(root / "workflow-status.json", dict(stage="error", action=stage, returncode=code))
            raise RuntimeError(stage)
    write(root / "workflow-status.json", dict(stage="numerical_complete_pending_review", step=4000))
