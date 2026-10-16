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
command = [
    sys.executable,
    "scripts/field_width_diagnostic.py",
    "evaluate",
    "--root",
    str(root),
]
tick = time.monotonic()
started = datetime.now(UTC).isoformat()
with (root / "driver-evaluate.log").open("w") as log:
    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
record = dict(
    started_at=started,
    finished_at=datetime.now(UTC).isoformat(),
    seconds=time.monotonic() - tick,
    returncode=result.returncode,
    command=command,
)
(root / "driver-evaluate.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record), flush=True)
sys.exit(result.returncode)
