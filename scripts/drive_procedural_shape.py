"""Run one bounded phase, recording the actual child exit code and elapsed time."""

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "phase",
    choices=(
        "preflight",
        "train",
        "evaluation",
        "stress",
        "scene512",
        "width-sweep",
        "ambiguity",
        "geometry",
    ),
)
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--until", type=int, choices=tuple(range(1000, 6001, 1000)))
args = parser.parse_args()
root = args.root.resolve()
scripts = dict(
    preflight="audit_procedural_shape.py",
    train="procedural_shape_study.py",
    evaluation="evaluate_procedural_shape.py",
    stress="evaluate_procedural_shape_rough.py",
    scene512="evaluate_procedural_shape_scenes.py",
    ambiguity="evaluate_procedural_shape_ambiguity.py",
    geometry="evaluate_procedural_shape_geometry.py",
    **{"width-sweep": "evaluate_procedural_shape_sweep.py"},
)
command = [sys.executable, "scripts/" + scripts[args.phase]]
name = args.phase
if args.phase == "train":
    assert args.until is not None
    command += ["train", "--until", str(args.until)]
    name += "-" + str(args.until)
command += ["--root", str(root)]
started = datetime.now(UTC).isoformat()
tick = time.monotonic()
with (root / f"driver-{name}.log").open("w") as log:
    with subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT) as child:
        while True:
            try:
                code = child.wait(timeout=45)
                break
            except subprocess.TimeoutExpired:
                progress = root / "training/progress.json"
                status = json.loads(progress.read_text()) if progress.exists() else {}
                print(
                    json.dumps(
                        dict(phase=name, elapsed=time.monotonic() - tick, step=status.get("step"))
                    ),
                    flush=True,
                )
record = dict(
    phase=args.phase,
    until=args.until,
    started_at=started,
    finished_at=datetime.now(UTC).isoformat(),
    returncode=code,
    seconds=time.monotonic() - tick,
    command=command,
)
(root / f"driver-{name}.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record), flush=True)
sys.exit(code)
