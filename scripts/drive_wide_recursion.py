"""Run the frozen wide-recursion diagnostic with an actual exit record."""

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
a = p.parse_args()
root = a.root.resolve()
started = datetime.now(UTC).isoformat()
tick = time.monotonic()
command = [sys.executable, "scripts/wide_recursion_diagnostic.py", "evaluate", "--root", str(root)]
with (root / "driver-evaluate.log").open("w") as log:
    with subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT) as child:
        while True:
            try:
                code = child.wait(timeout=45)
                break
            except subprocess.TimeoutExpired:
                print(
                    json.dumps(dict(phase="evaluate", elapsed=time.monotonic() - tick)), flush=True
                )
record = dict(
    started_at=started,
    finished_at=datetime.now(UTC).isoformat(),
    seconds=time.monotonic() - tick,
    returncode=code,
    command=command,
)
(root / "driver-evaluate.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record), flush=True)
raise SystemExit(code)
