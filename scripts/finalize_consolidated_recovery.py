"""Retain the observed parent interruption alongside the original numerical finalizer."""

import argparse
import subprocess
import sys
from pathlib import Path

from campaign_protocol import read, sha, write

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
r = p.parse_args().root.resolve()
plan = read(r / "diagnostics/recovery-plan.json")
assert sha(Path(__file__)) == plan["finalizer_sha256"]
record = read(r / "diagnostics/interruption-1.json")
assert record["parent_exec_returncode"] == 143 and record["child_returncode"] is None
assert record["last_logged_step"] == record["last_saved_step"] == 4000
for f, h in record["evidence"].items():
    assert sha(r / f) == h
subprocess.run(
    [sys.executable, "scripts/finalize_consolidated_geometry.py", "--root", str(r)], check=True
)
original = read(r / "completion.json")
write(r / "diagnostics/completion-before-recovery-binding.json", original)
original["interruptions"] = [record]
original["evidence"].update(
    {
        f: sha(r / f)
        for f in [
            "diagnostics/interruption-1.json",
            "diagnostics/recovery-plan.json",
            "diagnostics/completion-before-recovery-binding.json",
        ]
    }
)
write(r / "completion.json", original)
print(dict(complete=True, parent_interruptions=1, numerical_code_unchanged=True), flush=True)
