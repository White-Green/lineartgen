"""Verify fresh initialization and both portions of the 24k input trajectory."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from consolidated_geometry_study import (
    CONTROL,
    PARENT,
    arms_from_root,
    batch,
    load_data,
    replay_state,
)
from forced_geometry_study import arms_from_root as old_arms
from forced_geometry_study import batch as old_batch
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
old = old_arms(CONTROL)
reference, _ = load_field(PARENT / "initials/mixed")
early = [json.loads(s) for s in (PARENT / "training/sampling.jsonl").read_text().splitlines()]
late = [json.loads(s) for s in (CONTROL / "training/sampling.jsonl").read_text().splitlines()]
rows = []
for k in range(6):
    a = batch(data, k, "fresh")
    assert digest((a[0], a[1][0], a[2])) == early[k]["input_sha256"]["mixed"]
    pred = np.asarray(infer(new["fresh"]["model"], a[0]))
    np.testing.assert_array_equal(
        pred, np.asarray(infer_dilation(reference, a[0], mode="physical"))
    )
    parts, grads = new["fresh"]["gradient"](new["fresh"]["model"], *a)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves((parts, grads)))
    b = batch(data, 18000 + k, "fresh")
    c = old_batch({"supported": data}, k, "supported")
    exact(b, c)
    assert digest((b[0], b[1][0], b[2])) == late[k]["input_sha256"]["supported"]
    before = old["supported"]["gradient"](old["supported"]["model"], *c)
    exact(before, new["fresh"]["gradient"](old["supported"]["model"], *b))
    actual = update(old["supported"], *c)
    rows.append(
        dict(
            depth=k + 1,
            fresh_forward_exact=True,
            fresh_gradient_finite=True,
            late_inputs_targets_masks_exact=True,
            late_gradient_exact=True,
            metric_replay=replay_state(actual, late[k]["metrics"]["supported"]),
        )
    )
restored = old_arms(CONTROL)
with checkpoints.manager(CONTROL / "training/checkpoints", keep=8) as saved:
    assert lineart_state.restore(saved, restored, 6) == 6
state_replay = replay_state(
    lineart_state.payload(old, 6)["arms"]["supported"],
    lineart_state.payload(restored, 6)["arms"]["supported"],
)
for name in (
    "forced_context_objective",
    "quartic_field_objective",
    "context_dropout_objective",
    "joint_geometry_study",
    "physical_field_model",
    "group_field_model",
    "subpixel_field_model",
    "support_merge_data",
    "scale_style_balance_study",
):
    f = Path("scripts") / (name + ".py")
    assert sha(f) == sha(CONTROL.parent.parent / f)
write(
    root / "preflight.json",
    dict(
        passed=True,
        rows=rows,
        control_full_state_replay=state_replay,
        inherited_tests=read(CONTROL / "test-results.json"),
    ),
)
print(dict(passed=True, state_replay=state_replay), flush=True)
