"""Audit fixed initial weights, unchanged control updates and local gradients."""

import argparse
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from cue_capacity import make_arm
from forced_context_objective import forced_gradient
from local_sign_study import CONTROL, METHODS, arms_from_root, batch, check, load_data, replay_state
from physical_field_model import infer, load_field
from plausible_study import exact
from scale_style_balance_study import update

from lineartgen.training import lineart_state

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
root = p.parse_args().root.resolve()
check(root)
assert jax.devices()[0].platform != "cpu"
data, arms = load_data(root), arms_from_root(root)
reference, _ = load_field(CONTROL / "snapshots/step-24000/aligned/raw")
old = make_arm(arms["global"]["config"], reference)
old["gradient"] = forced_gradient()
rows = []
for k in range(6):
    b = batch(data, k, "global")
    exact(b, batch(data, k, "local"))
    for method in METHODS:
        # Each later check uses an untouched initial clone, not the updated control.
        initial, _ = load_field(root / f"initials/{method}")
        parent, _ = load_field(root / "reference")
        exact(infer(initial, b[0]), infer(parent, b[0]))
        parts, g = arms[method]["gradient"](arms[method]["model"], *b)
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves((parts, g)))
        assert max(float(abs(np.asarray(x)).max()) for x in jax.tree.leaves(g)) > 0
    actual = update(arms["global"], *b)
    expected = update(old, *b)
    rows.append(
        dict(
            depth=k + 1,
            initial_prediction_exact=True,
            gradients_finite_nonzero=True,
            control_metrics=replay_state(actual, expected),
            teacher_audit=data["last_audit"],
        )
    )
state = replay_state(
    lineart_state.payload({"global": arms["global"]}, 6)["arms"]["global"],
    lineart_state.payload({"global": old}, 6)["arms"]["global"],
)
batch(data, 5999, "global")
late = data["last_audit"].copy()
review = read(root / "teacher-preview/visual-review.json")
for f, h in review["files"].items():
    assert sha(root / "teacher-preview" / f) == h == sha(CONTROL / "teacher-preview" / f)
write(
    root / "preflight.json",
    dict(
        passed=True,
        rows=rows,
        control_full_state_replay=state,
        final_batch_audit=late,
        teacher_preview_inherited=True,
        teacher_preview_sha256=sha(root / "teacher-preview/visual-review.json"),
    ),
)
print(dict(preflight=True, control_full_state_replay=state), flush=True)
