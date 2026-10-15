"""Run one bounded phase, recording the actual child exit code and elapsed time."""

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("phase", choices=("train", "evaluation", "stress", "scene512", "width-sweep"))
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--until", type=int, choices=(3000, 6000))
args = parser.parse_args()
root = args.root.resolve()
scripts = dict(
    train="width_coverage_study.py",
    evaluation="evaluate_width_coverage.py",
    stress="evaluate_width_coverage_rough.py",
    scene512="evaluate_width_coverage_scenes.py",
    **{"width-sweep": "evaluate_width_coverage_sweep.py"},
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
    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
record = dict(
    phase=args.phase,
    until=args.until,
    started_at=started,
    finished_at=datetime.now(UTC).isoformat(),
    returncode=result.returncode,
    seconds=time.monotonic() - tick,
    command=command,
)
(root / f"driver-{name}.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record), flush=True)
sys.exit(result.returncode)
