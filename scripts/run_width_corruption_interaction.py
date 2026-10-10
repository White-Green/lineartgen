"""Run the frozen width/corruption study, retaining the six-step startup state."""

import fcntl
import subprocess
import sys
from pathlib import Path

from campaign_protocol import read, write
from train_width_corruption_interaction import check

root = Path("runs/width-corruption-interaction-20261010").resolve()
with (root / "coordinator.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = check(root)
    gate = Path(plan["width_gate"])
    assert read(gate / "final-audit.json")["stage"] == "complete_diagnostic_insufficient"
    assert read(root / "data-audit.json")["crop_resume_exact"]
    assert read(root / "test-results.json")["passed"]
    commands = [
        ("startup6", ["scripts/train_width_corruption_interaction.py", "train", "--end", "6"]),
    ]
    for step in (1000, 4000):
        commands.extend(
            [
                (
                    f"train{step}",
                    ["scripts/train_width_corruption_interaction.py", "train", "--end", str(step)],
                ),
                (f"local{step}", ["scripts/evaluate_procedural_structure.py", "--step", str(step)]),
                (
                    f"control{step}",
                    [
                        "scripts/train_width_corruption_interaction.py",
                        "audit-broad",
                        "--end",
                        str(step),
                    ],
                ),
            ]
        )
    commands.extend(
        [
            ("development", ["scripts/evaluate_width_corruption_interaction.py", "evaluate"]),
            ("audit", ["scripts/train_width_corruption_interaction.py", "audit"]),
        ]
    )
    for stage, args in commands:
        marker = root / f"{stage}-complete.json"
        if marker.exists():
            assert read(marker)["completed"]
            continue
        write(root / "workflow-status.json", dict(stage=stage))
        with (root / f"{stage}.log").open("a") as log:
            code = subprocess.run(
                [sys.executable, *args, "--root", str(root)], stdout=log, stderr=subprocess.STDOUT
            ).returncode
        if code:
            write(root / "workflow-status.json", dict(stage="error", action=stage, returncode=code))
            raise RuntimeError(stage)
        write(marker, dict(completed=True, stage=stage))
    write(root / "workflow-status.json", dict(stage="numerical_complete_pending_review", step=4000))
