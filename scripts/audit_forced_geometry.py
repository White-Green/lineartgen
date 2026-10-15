"""Verify six-depth family-selection equivalence and replay the EY trajectory."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from forced_context_objective import forced_gradient
from forced_geometry_study import CONTROL, arms_from_root, batch, load_data, replay_state
from physical_field_model import infer, load_field
from plausible_study import digest, exact
from quartic_sign_study import arms_from_root as control_arms
from quartic_sign_study import batch as control_batch
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
legacy = forced_gradient(force_merge=False)
rows = []
for step in range(6):
    a, b = batch(data, step, "forced"), control_batch(data["forced"], step, "quartic")
    exact(a, b)
    assert digest((a[0], a[1][0], a[2])) == old_rows[step]["input_sha256"]["quartic"]
    actual = np.asarray(infer(new["forced"]["model"], a[0]))
    expected = np.asarray(infer_dilation(reference, a[0], mode="physical"))
    np.testing.assert_array_equal(actual, expected)
    old_model = old["quartic"]["model"]
    before = old["quartic"]["gradient"](old_model, *b)
    exact(before, legacy(old_model, *a))
    for method in ("forced", "supported"):
        candidate = batch(data, step, method)
        exact(candidate[0], a[0])
        exact(candidate[1][0], a[1][0])
        exact(candidate[1][3], a[1][3])
        parts4, grads4 = new[method]["gradient"](old_model, *candidate)
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves((parts4, grads4)))
        assert float(parts4["merged_fraction"]) == np.asarray(candidate[1][2]).mean()
    parts = update(old["quartic"], *b)
    rows.append(
        dict(
            depth=step + 1,
            diagnostic_forward_exact=True,
            legacy_selection_gradient_exact=True,
            forced_finite=True,
            legacy_loss=float(before[0]["loss"]),
            forced_loss=float(parts4["loss"]),
            control_metric_replay=replay_state(parts, old_rows[step]["metrics"]["quartic"]),
        )
    )
restored = control_arms(CONTROL)
with checkpoints.manager(CONTROL / "training/checkpoints", keep=8) as saved:
    assert lineart_state.restore(saved, restored, 6) == 6
actual = lineart_state.payload(old, 6)["arms"]["quartic"]
expected = lineart_state.payload(restored, 6)["arms"]["quartic"]
state_replay = replay_state(actual, expected)
inherited = read(CONTROL / "test-results.json")
assert inherited["passed"]
for name in (
    "context_dropout_objective",
    "joint_geometry_study",
    "merge_field_objective",
    "regional_field_objective",
    "quartic_field_objective",
    "quartic_context_objective",
    "support_merge_data",
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
        inputs_original_targets_masks_exact=True,
        six_depth_legacy_selection_gradients_exact=True,
        control_full_state_replay=state_replay,
        inherited_tests=inherited,
    ),
)
print(dict(preflight=True, state_replay=state_replay), flush=True)
