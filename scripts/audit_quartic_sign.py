"""Verify six-depth RMS equivalence, L4 gradients and the reused EX trajectory."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from physical_dilation_study import arms_from_root as control_arms
from physical_dilation_study import batch as control_batch
from physical_field_model import infer, load_field
from plausible_study import digest, exact
from quartic_context_objective import quartic_gradient
from quartic_sign_study import CONTROL, arms_from_root, batch, load_data, replay_state
from scale_dilation_model import infer_dilation
from scale_style_balance_study import update

from lineartgen.training import checkpoints, lineart_state

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
root = p.parse_args().root.resolve()
assert jax.devices()[0].platform != "cpu"
data = load_data(root)
new, old = arms_from_root(root), control_arms(CONTROL)
reference, _ = load_field(root / "reference")
old_rows = [json.loads(x) for x in (CONTROL / "training/sampling.jsonl").read_text().splitlines()]
rms = quartic_gradient(sign_power=2)
rows = []
for step in range(6):
    a, b = batch(data, step, "quartic"), control_batch(data, step, "physical")
    exact(a, b)
    assert digest((a[0], a[1][0], a[2])) == old_rows[step]["input_sha256"]["physical"]
    actual = np.asarray(infer(new["quartic"]["model"], a[0]))
    expected = np.asarray(infer_dilation(reference, a[0], mode="physical"))
    np.testing.assert_array_equal(actual, expected)
    old_model = old["physical"]["model"]
    before = old["physical"]["gradient"](old_model, *b)
    exact(before, rms(old_model, *a))
    parts4, grads4 = new["quartic"]["gradient"](old_model, *a)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves((parts4, grads4)))
    parts = update(old["physical"], *b)
    rows.append(
        dict(
            depth=step + 1,
            diagnostic_forward_exact=True,
            rms_gradient_exact=True,
            quartic_finite=True,
            rms_loss=float(before[0]["loss"]),
            quartic_loss=float(parts4["loss"]),
            control_metric_replay=replay_state(parts, old_rows[step]["metrics"]["physical"]),
        )
    )
restored = control_arms(CONTROL)
with checkpoints.manager(CONTROL / "training/checkpoints", keep=8) as saved:
    assert lineart_state.restore(saved, restored, 6) == 6
actual = lineart_state.payload(old, 6)["arms"]["physical"]
expected = lineart_state.payload(restored, 6)["arms"]["physical"]
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
    "physical_field_model",
    "scale_dilation_model",
):
    file = Path("scripts") / (name + ".py")
    assert sha(file) == sha(CONTROL.parent.parent / file)
write(
    root / "preflight.json",
    dict(
        passed=True,
        rows=rows,
        inputs_targets_masks_exact=True,
        six_depth_rms_gradients_exact=True,
        control_full_state_replay=state_replay,
        inherited_tests=inherited,
    ),
)
print(dict(preflight=True, state_replay=state_replay), flush=True)
