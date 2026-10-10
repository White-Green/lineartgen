"""Continue the frozen AE parent; completed stage markers permit safe coordinator restart."""

import fcntl
import subprocess
import sys
from pathlib import Path

from campaign_protocol import read, write
from train_paper_band_long_run import check

root = Path("runs/paper-band-long-run-20261010").resolve()
with (root / "coordinator.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = check(root)
    assert read(root / "data-audit.json")["crop_resume_exact"]
    assert read(root / "test-results.json")["passed"]
    commands = [
        ("startup4006", ["scripts/train_paper_band_long_run.py", "train", "--end", "4006"]),
        ("startup-replay", ["scripts/audit_paper_band_replay.py"]),
        ("local4000", ["scripts/evaluate_procedural_structure.py", "--step", "4000"]),
        ("control4000", ["scripts/train_paper_band_long_run.py", "audit-broad", "--end", "4000"]),
    ]
    for step in (8000, 16000):
        commands.extend(
            [
                (
                    f"train{step}",
                    ["scripts/train_paper_band_long_run.py", "train", "--end", str(step)],
                ),
                (f"local{step}", ["scripts/evaluate_procedural_structure.py", "--step", str(step)]),
                (
                    f"control{step}",
                    ["scripts/train_paper_band_long_run.py", "audit-broad", "--end", str(step)],
                ),
            ]
        )
    commands.extend(
        [
            ("development", ["scripts/evaluate_paper_band_long_run.py", "evaluate"]),
            ("audit", ["scripts/train_paper_band_long_run.py", "audit"]),
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
    write(
        root / "workflow-status.json", dict(stage="numerical_complete_pending_review", step=16000)
    )
