"""Add interrupted audit/training evidence to the unchanged numerical finalizer."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from campaign_protocol import read, sha, write
from widened_geometry_study import replay_state

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
new_rows = [json.loads(x) for x in (r / "training/sampling.jsonl").read_text().splitlines()]
interruptions = []
replays = []
for path in plan["training_interruptions"]:
    interruption = read(r / path)
    assert (
        interruption["parent_exec_returncode"] == 143 and interruption["child_returncode"] is None
    )
    for f, h in interruption["evidence"].items():
        assert sha(r / f) == h
    journal = next(f for f in interruption["evidence"] if f.endswith("training--sampling.jsonl"))
    old_rows = [json.loads(x) for x in (r / journal).read_text().splitlines()]
    start, end = interruption["last_saved_step"], interruption["last_logged_step"]
    assert len(old_rows) == end and end >= start
    checks = []
    for old, new in zip(old_rows[start:], new_rows[start:end], strict=True):
        for key in old:
            if key not in ("seconds", "metrics"):
                assert old[key] == new[key]
        checks.append(replay_state(new["metrics"], old["metrics"]))
    replays.append(
        dict(
            interruption=path,
            journaled_updates=end - start,
            inputs_targets_masks_exact=True,
            max_metric_difference=max((x["max_abs_difference"] for x in checks), default=0),
            metrics_exact=all(x["exact"] for x in checks),
            unsaved_parameter_states_compared=False,
        )
    )
    interruptions.append(interruption)
write(r / "diagnostics/interrupted-prefix-replay.json", replays)
subprocess.run(
    [sys.executable, "scripts/finalize_widened_geometry.py", "--root", str(r)], check=True
)
c = read(r / "completion.json")
write(r / "diagnostics/completion-before-recovery-binding.json", c)
c["preflight_interruptions"] = [a]
c["preflight_retry"] = b
c["training_interruptions"] = interruptions
c["interrupted_prefix_replay"] = replays
files = [
    "diagnostics/preflight-attempt1.json",
    "diagnostics/preflight-attempt2.json",
    "diagnostics/preflight-attempt2.log",
    "diagnostics/recovery-plan.json",
    "diagnostics/interrupted-prefix-replay.json",
    "diagnostics/completion-before-recovery-binding.json",
]
files += plan["training_interruptions"]
c["evidence"].update({f: sha(r / f) for f in files})
write(r / "completion.json", c)
print(
    dict(
        complete=True,
        preflight_interruptions=1,
        training_interruptions=len(interruptions),
        numerical_code_unchanged=True,
    ),
    flush=True,
)
