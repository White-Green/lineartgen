"""Record bounded phase status and actual exit code."""

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("phase", choices=("train", "evaluate"))
p.add_argument("--root", type=Path, required=True)
a = p.parse_args()
r = a.root.resolve()
command = [sys.executable, "scripts/axis_replay_mix.py", a.phase, "--root", str(r)]
tick = time.monotonic()
started = datetime.now(UTC).isoformat()
with (r / f"driver-{a.phase}.log").open("w") as f:
    with subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT) as child:
        while True:
            try:
                code = child.wait(timeout=45)
                break
            except subprocess.TimeoutExpired:
                status = r / "training/progress.json"
                step = json.loads(status.read_text()).get("step") if status.exists() else None
                print(dict(phase=a.phase, step=step, elapsed=time.monotonic() - tick), flush=True)
record = dict(
    phase=a.phase,
    returncode=code,
    seconds=time.monotonic() - tick,
    started_at=started,
    finished_at=datetime.now(UTC).isoformat(),
    command=command,
)
(r / f"driver-{a.phase}.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record), flush=True)
sys.exit(code)
