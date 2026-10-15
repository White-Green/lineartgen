"""Audit function-preserving widening, finite gradients, and inherited optimizer replay."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import sha, write
from context_dropout_objective import dropout_forward
from forced_geometry_study import arms_from_root as old_arms
from forced_geometry_study import batch as old_batch
from physical_field_model import infer, load_field
from plausible_study import digest, exact
from scale_style_balance_study import update
from widened_geometry_study import CONTROL, arms_from_root, batch, check, load_data, replay_state

from lineartgen.training import checkpoints, lineart_state

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
root = p.parse_args().root.resolve()
check(root)
assert jax.devices()[0].platform != "cpu"
data = load_data(root)
new = arms_from_root(root)
old = old_arms(CONTROL)
reference, _ = load_field(root / "reference")
late = [json.loads(x) for x in (CONTROL / "training/sampling.jsonl").read_text().splitlines()]
rows = []
for k in range(6):
    a = batch(data, k, "control")
    b = batch(data, k, "wide")
    exact(a, b)
    small = new["control"]["model"]
    wide = new["wide"]["model"]
    exact(infer(small, a[0]), infer(reference, a[0]))
    predicted = replay_state(infer(wide, a[0]), infer(small, a[0]))
    fields = replay_state(
        dropout_forward(wide, a[0], a[1][3]), dropout_forward(small, a[0], a[1][3])
    )
    gradients = {}
    for m in new:
        parts, g = new[m]["gradient"](new[m]["model"], *a)
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves((parts, g)))
        gradients[m] = dict(loss=float(parts["loss"]), finite=True)
        if m == "wide":
            maximum = float(abs(np.asarray(g.head.linear.kernel[...][6:])).max())
            assert maximum > 0
            gradients[m]["new_head_gradient_max"] = maximum
    c = old_batch({"supported": data}, k, "supported")
    assert digest((c[0], c[1][0], c[2])) == late[k]["input_sha256"]["supported"]
    parts = update(old["supported"], *c)
    rows.append(
        dict(
            depth=k + 1,
            initial_control_exact=True,
            initial_prediction=predicted,
            initial_fields=fields,
            identical_batch=True,
            gradients=gradients,
            inherited_metric_replay=replay_state(parts, late[k]["metrics"]["supported"]),
        )
    )
restored = old_arms(CONTROL)
with checkpoints.manager(CONTROL / "training/checkpoints", keep=8) as saved:
    assert lineart_state.restore(saved, restored, 6) == 6
state = replay_state(
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
write(root / "preflight.json", dict(passed=True, rows=rows, control_full_state_replay=state))
print(dict(preflight=True, control_full_state_replay=state), flush=True)
