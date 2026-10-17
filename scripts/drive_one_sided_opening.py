"""Run the frozen one-sided opening diagnostic with an actual exit record."""

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("phase", choices=("train-400", "train-800", "train-1200", "evaluate"))
p.add_argument("--root", type=Path, required=True)
a = p.parse_args()
root = a.root.resolve()
started = datetime.now(UTC).isoformat()
tick = time.monotonic()
command = [
    sys.executable,
    "scripts/one_sided_opening_fit.py",
    "train" if a.phase.startswith("train-") else a.phase,
    "--root",
    str(root),
]
if a.phase.startswith("train-"):
    command += ["--until", a.phase.split("-")[1]]
with (root / f"driver-{a.phase}.log").open("w") as log:
    with subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT) as child:
        while True:
            try:
                code = child.wait(timeout=45)
                break
            except subprocess.TimeoutExpired:
                print(json.dumps(dict(phase=a.phase, elapsed=time.monotonic() - tick)), flush=True)
record = dict(
    started_at=started,
    finished_at=datetime.now(UTC).isoformat(),
    seconds=time.monotonic() - tick,
    returncode=code,
    command=command,
)
(root / f"driver-{a.phase}.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record), flush=True)
raise SystemExit(code)
