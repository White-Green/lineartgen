"""Prepare final raw/EMA exports after training, without selecting or publishing weights."""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def now():
    return datetime.datetime.now(datetime.UTC).isoformat()


def prepare(root, parent, wait):
    state_path = root / "delivery-preparation.json"
    if state_path.exists():
        raise FileExistsError(
            "Delivery preparation already has a record; inspect it before retrying"
        )
    state = dict(started_at=now(), phase="waiting", operations=[], published=False)

    def save():
        state_path.write_text(json.dumps(state, indent=2) + "\n")

    save()
    while not (root / "training-summary.json").exists():
        if not wait:
            raise RuntimeError("Training has not completed")
        time.sleep(30)
    steps = json.loads((root / "plan.json").read_text())["added_steps"]
    env = os.environ | dict(
        PYTHONPATH="src:scripts:packages/lineartgen-runtime/src",
        JAX_PLATFORMS="cpu",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )

    def run(phase, script, *arguments):
        command = [sys.executable, "scripts/" + script, *map(str, arguments)]
        row = dict(phase=phase, command=command, started_at=now())
        state.update(phase=phase)
        state["operations"].append(row)
        save()
        before = time.monotonic()
        with (root / f"delivery-{phase}.log").open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
        row.update(
            returncode=result.returncode, seconds=time.monotonic() - before, finished_at=now()
        )
        save()
        if result.returncode:
            state.update(phase="failed", failed_operation=phase)
            save()
            raise RuntimeError(f"{phase} failed; see delivery-{phase}.log")

    run("audit", "audit_graph_training.py", "--root", root)
    exports = {}
    for mode in ("raw", "ema"):
        exports[mode] = root.parent / f"graph-export-final-{mode}"
        if exports[mode].exists():
            raise FileExistsError(exports[mode])
        run(
            f"export-{mode}",
            "export_graph_model.py",
            "--root",
            root,
            "--checkpoint",
            root / f"snapshots/step-{steps}/{mode}.npz",
            "--destination",
            exports[mode],
            "--total-steps",
            steps + 16200,
        )
    run(
        "comparison",
        "compare_graph_checkpoints.py",
        "--candidate",
        "raw=" + str(exports["raw"]),
        "--candidate",
        "ema=" + str(exports["ema"]),
        "--inputs",
        parent,
        "--output",
        root.parent / "graph-final-comparison",
    )
    state.update(phase="awaiting_visual_review", finished_at=now(), prepared=True)
    save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    prepare(**vars(parser.parse_args()))
