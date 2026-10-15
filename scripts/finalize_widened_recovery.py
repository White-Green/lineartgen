"""Add the interrupted preflight evidence to the unchanged numerical finalizer."""

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
a = read(r / "diagnostics/preflight-attempt1.json")
b = read(r / "diagnostics/preflight-attempt2.json")
assert a["exec_returncode"] == 143 and a["training_started"] is False
assert b["child_returncode"] == 0
for f, h in a["evidence"].items():
    assert sha(r / f) == h
subprocess.run(
    [sys.executable, "scripts/finalize_widened_geometry.py", "--root", str(r)], check=True
)
c = read(r / "completion.json")
write(r / "diagnostics/completion-before-recovery-binding.json", c)
c["preflight_interruptions"] = [a]
c["preflight_retry"] = b
c["evidence"].update(
    {
        f: sha(r / f)
        for f in [
            "diagnostics/preflight-attempt1.json",
            "diagnostics/preflight-attempt2.json",
            "diagnostics/preflight-attempt2.log",
            "diagnostics/recovery-plan.json",
            "diagnostics/completion-before-recovery-binding.json",
        ]
    }
)
write(r / "completion.json", c)
print(dict(complete=True, preflight_interruptions=1, numerical_code_unchanged=True), flush=True)
