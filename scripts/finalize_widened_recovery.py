"""Add the interrupted preflight evidence to the unchanged numerical finalizer."""

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
interruption = read(r / "diagnostics/training-interruption-1.json")
assert interruption["parent_exec_returncode"] == 143 and interruption["child_returncode"] is None
assert interruption["last_saved_step"] == 1000 and interruption["last_logged_step"] == 1042
for f, h in interruption["evidence"].items():
    assert sha(r / f) == h
old_rows = [
    json.loads(x)
    for x in (r / "diagnostics/training-interruption-1/training--sampling.jsonl")
    .read_text()
    .splitlines()
]
new_rows = [json.loads(x) for x in (r / "training/sampling.jsonl").read_text().splitlines()]
replays = []
for old, new in zip(old_rows[1000:], new_rows[1000:1042], strict=True):
    for key in old:
        if key not in ("seconds", "metrics"):
            assert old[key] == new[key]
    replays.append(replay_state(new["metrics"], old["metrics"]))
replay = dict(
    journaled_updates=42,
    inputs_targets_masks_exact=True,
    max_metric_difference=max(x["max_abs_difference"] for x in replays),
    metrics_exact=all(x["exact"] for x in replays),
    unsaved_parameter_states_compared=False,
)
write(r / "diagnostics/interrupted-prefix-replay.json", replay)
subprocess.run(
    [sys.executable, "scripts/finalize_widened_geometry.py", "--root", str(r)], check=True
)
c = read(r / "completion.json")
write(r / "diagnostics/completion-before-recovery-binding.json", c)
c["preflight_interruptions"] = [a]
c["preflight_retry"] = b
c["training_interruptions"] = [interruption]
c["interrupted_prefix_replay"] = replay
c["evidence"].update(
    {
        f: sha(r / f)
        for f in [
            "diagnostics/training-interruption-1.json",
            "diagnostics/interrupted-prefix-replay.json",
            "diagnostics/preflight-attempt1.json",
            "diagnostics/preflight-attempt2.json",
            "diagnostics/preflight-attempt2.log",
            "diagnostics/recovery-plan.json",
            "diagnostics/completion-before-recovery-binding.json",
        ]
    }
)
write(r / "completion.json", c)
print(
    dict(
        complete=True,
        preflight_interruptions=1,
        training_interruptions=1,
        numerical_code_unchanged=True,
    ),
    flush=True,
)
