"""Verify new forward/serialization, unchanged data and the reused ET dropout trajectory."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from context_dropout_study import arms_from_root as control_arms
from context_dropout_study import batch as control_batch
from physical_dilation_study import CONTROL, arms_from_root, batch, load_data, replay_state
from physical_field_model import infer, load_field
from plausible_study import digest, exact
from scale_dilation_model import infer_dilation
from scale_style_balance_study import update

from lineartgen.training import checkpoints, lineart_state

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
root = p.parse_args().root.resolve()
assert jax.devices()[0].platform != "cpu"
data = load_data(root)
new = arms_from_root(root)
old = control_arms(CONTROL)
reference, _ = load_field(root / "reference")
old_rows = [json.loads(x) for x in (CONTROL / "training/sampling.jsonl").read_text().splitlines()]
rows = []
for step in range(6):
    a, b = batch(data, step, "physical"), control_batch(data, step, "dropout")
    exact(a, b)
    assert digest((a[0], a[1][0], a[2])) == old_rows[step]["input_sha256"]["dropout"]
    actual = np.asarray(infer(new["physical"]["model"], a[0]))
    expected = np.asarray(infer_dilation(reference, a[0], mode="physical"))
    np.testing.assert_array_equal(actual, expected)
    if step == 0:
        exact(
            new["physical"]["gradient"](new["physical"]["model"], *a),
            old["dropout"]["gradient"](old["dropout"]["model"], *b),
        )
    parts = update(old["dropout"], *b)
    rows.append(
        dict(
            depth=step + 1,
            diagnostic_forward_exact=True,
            control_metric_replay=replay_state(parts, old_rows[step]["metrics"]["dropout"]),
        )
    )
restored = control_arms(CONTROL)
with checkpoints.manager(CONTROL / "training/checkpoints", keep=4) as saved:
    assert lineart_state.restore(saved, restored, 6) == 6
actual = lineart_state.payload(old, 6)["arms"]["dropout"]
expected = lineart_state.payload(restored, 6)["arms"]["dropout"]
state_replay = replay_state(actual, expected)
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
        inputs_targets_masks_exact=True,
        depth1_gradient_exact=True,
        control_full_state_replay=state_replay,
        inherited_tests=inherited,
    ),
)
print(dict(preflight=True, state_replay=state_replay), flush=True)
