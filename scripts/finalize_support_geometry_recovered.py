"""Validate the interrupted continuation replay and retain its nonzero exit record."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from campaign_protocol import read, sha, write

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
root = p.parse_args().root.resolve()
d = root / "diagnostics/recovery"
plan = read(d / "plan.json")
assert plan["exec_exit_code"] == 143 and plan["restart_checkpoint"] == 3000
for name, h in plan["bindings"].items():
    assert sha(d / name) == h
for file, h in read(d / "validator-plan.json")["files"].items():
    assert sha(Path(file)) == h
old = [json.loads(x) for x in (d / "attempt-1-sampling.jsonl").read_text().splitlines()]
new = [json.loads(x) for x in (root / "training/sampling.jsonl").read_text().splitlines()]
assert len(new) == 6000 and len(old) == plan["last_journal_step"]
maximum = 0.0
exact_metrics = True
for a, b in zip(old, new[: len(old)], strict=True):
    if a["step"] <= 3000:
        assert a == b
    else:
        assert {k: v for k, v in a.items() if k not in ("seconds", "metrics")} == {
            k: v for k, v in b.items() if k not in ("seconds", "metrics")
        }
        for m in a["metrics"]:
            assert a["metrics"][m].keys() == b["metrics"][m].keys()
            for k, x in a["metrics"][m].items():
                y = b["metrics"][m][k]
                np.testing.assert_allclose(x, y, atol=plan["metric_atol"], rtol=plan["metric_rtol"])
                maximum = max(maximum, abs(x - y))
                exact_metrics = exact_metrics and x == y
write(
    d / "replay.json",
    dict(
        passed=True,
        replayed_steps=len(old) - 3000,
        max_abs_metric_difference=maximum,
        exact_metrics=exact_metrics,
        inputs_and_candidates_exact=True,
    ),
)
subprocess.run(
    [sys.executable, "scripts/finalize_support_geometry.py", "--root", str(root)], check=True
)
c = read(root / "completion.json")
c["interrupted_attempt"] = dict(
    exec_exit_code=143,
    cause="unknown",
    restart_checkpoint=3000,
    retained=True,
    replay=read(d / "replay.json"),
)
c["evidence"].update({str(p.relative_to(root)): sha(p) for p in d.iterdir() if p.is_file()})
write(root / "completion.json", c)
print(dict(complete_with_recorded_recovery=True, replay=read(d / "replay.json")), flush=True)
