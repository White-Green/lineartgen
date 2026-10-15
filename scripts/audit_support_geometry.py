"""Verify reused control replay and six-depth equivalence before support-gate training."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from context_dropout_study import arms_from_root as control_arms
from joint_geometry_study import load_data as control_data
from merge_field_objective import merged_gradient
from plausible_study import digest, exact
from scale_style_balance_study import update
from support_geometry_study import (
    CONTROL,
    PARENT,
    arms_from_root,
    batch,
    load_data,
    replay_state,
)

from lineartgen.training import checkpoints, lineart_state

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
root = p.parse_args().root.resolve()
assert jax.devices()[0].platform != "cpu"
new_data, old_data = load_data(root), control_data(CONTROL)
new, old = arms_from_root(root), arms_from_root(root)
old_rows = [json.loads(x) for x in (CONTROL / "training/sampling.jsonl").read_text().splitlines()]
plain_gradient = merged_gradient()
rows = []
for step in range(6):
    a, b = batch(old_data, step, "control"), batch(new_data, step, "support")
    exact(a[0], b[0])
    exact(a[1][0], b[1][0])
    exact(a[2], b[2])
    assert digest((a[0], a[1][0], a[2])) == old_rows[step]["input_sha256"]["control"]
    old_flag, new_flag = np.asarray(a[1][2]), np.asarray(b[1][2])
    assert np.all(~old_flag | new_flag)
    grads = old["support"]["gradient"](old["support"]["model"], *a)
    plain = plain_gradient(old["support"]["model"], a[0], a[1][:3], a[2])
    exact(grads, plain)
    if np.array_equal(old_flag, new_flag):
        exact(a, b)
        exact(grads, new["support"]["gradient"](old["support"]["model"], *b))
    parts = update(old["support"], *a)
    replay = replay_state(parts, old_rows[step]["metrics"]["control"])
    rows.append(
        dict(
            depth=step + 1,
            unchanged_flags=bool(np.array_equal(old_flag, new_flag)),
            plain_gradient_exact=True,
            control_metric_replay=replay,
        )
    )
restored = control_arms(CONTROL)
with checkpoints.manager(CONTROL / "training/checkpoints", keep=4) as saved:
    assert lineart_state.restore(saved, restored, 6) == 6
actual = lineart_state.payload(old, 6)["arms"]["support"]
expected = lineart_state.payload(restored, 6)["arms"]["control"]
state_replay = replay_state(actual, expected)
cache = read(root / "merge-cache.json")
assert cache["old_fields_exact"] and cache["old_fields"] == 27
inherited = read(CONTROL / "test-results.json")
assert inherited["passed"]
for name in (
    "context_dropout_objective",
    "joint_geometry_study",
    "merge_field_objective",
    "regional_field_objective",
    "group_field_model",
    "subpixel_field_model",
):
    file = Path("scripts") / (name + ".py")
    assert sha(file) == sha(CONTROL.parent.parent / file)
write(
    root / "preflight.json",
    dict(
        passed=True,
        rows=rows,
        control_full_state_replay=state_replay,
        inherited_tests=inherited,
        cache=cache,
        same_parent=str(PARENT),
    ),
)
print(dict(preflight=True, state_replay=state_replay), flush=True)
