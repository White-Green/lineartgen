"""Record actual evaluation child status and elapsed wall time."""

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
root = p.parse_args().root.resolve()
command = [sys.executable, "scripts/field_repair_study.py", "evaluate", "--root", str(root)]
tick = time.monotonic()
started = datetime.now(UTC).isoformat()
with (root / "driver-evaluate.log").open("w") as log:
    with subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT) as child:
        while True:
            try:
                code = child.wait(timeout=45)
                break
            except subprocess.TimeoutExpired:
                print(dict(phase="field-repair", elapsed=time.monotonic() - tick), flush=True)
record = dict(
    started_at=started,
    finished_at=datetime.now(UTC).isoformat(),
    seconds=time.monotonic() - tick,
    returncode=code,
    command=command,
)
(root / "driver-evaluate.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record), flush=True)
sys.exit(code)
