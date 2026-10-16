"""Check identical source batches, neutral weighting and untouched control updates."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from campaign_protocol import read, sha, write
from cue_capacity import make_arm
from opening_prior_objective import opening_gradient
from physical_field_model import infer, load_field
from plausible_study import exact
from scale_style_balance_study import update
from wide_emphasis_study import (
    METHODS,
    SOURCE,
    arms_from_root,
    batch,
    check,
    load_data,
    replay_state,
)

from lineartgen.training import lineart_state

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
root = p.parse_args().root.resolve()
check(root)
assert jax.devices()[0].platform != "cpu"
data, arms = load_data(root), arms_from_root(root)
reference, _ = load_field(SOURCE / "snapshots/step-6000/expanded/raw")
old = make_arm(arms["uniform"]["config"], reference)
old["gradient"] = opening_gradient()
rows = []
for k in range(6):
    b = batch(data, k, "uniform")
    candidate = batch(data, k, "emphasized")
    exact(b, candidate)
    initial, _ = load_field(root / "initials/emphasized")
    parent, _ = load_field(SOURCE / "snapshots/step-6000/expanded/raw")
    exact(infer(initial, b[0]), infer(parent, b[0]))
    for method in METHODS:
        parts, g = arms[method]["gradient"](arms[method]["model"], *b)
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves((parts, g)))
        assert max(float(abs(np.asarray(x)).max()) for x in jax.tree.leaves(g)) > 0
    neutral = (b[0], (*b[1][:4], jnp.ones((4,))), b[2])
    new_parts, new_grad = arms["emphasized"]["gradient"](initial, *neutral)
    old_parts, old_grad = old["gradient"](initial, b[0], b[1][:4], b[2])
    neutral_audit = replay_state((new_parts, new_grad), (old_parts, old_grad))
    actual = update(arms["uniform"], *b)
    expected = update(old, b[0], b[1][:4], b[2])
    rows.append(
        dict(
            depth=k + 1,
            inputs_exact=True,
            initial_exact=True,
            gradients_finite_nonzero=True,
            neutral_weight_parity=neutral_audit,
            control_metrics=replay_state(actual, expected),
            teacher_audit=data["last_audit"].copy(),
        )
    )
state = replay_state(
    lineart_state.payload({"uniform": arms["uniform"]}, 6)["arms"]["uniform"],
    lineart_state.payload({"uniform": old}, 6)["arms"]["uniform"],
)
checks = []
for k in range(100):
    b = batch(data, k, "uniform")
    exact(b, batch(data, k, "emphasized"))
    if any(data["last_audit"]["changed_examples"]):
        checks.append(dict(step=k, audit=data["last_audit"].copy()))
        if len(checks) == 4:
            break
assert len(checks) == 4
batch(data, 5999, "uniform")
late = data["last_audit"].copy()
for directory in ("teacher-preview", "source-preview"):
    review = read(root / directory / "visual-review.json")
    assert review["reviewed"]
    for f, h in review["files"].items():
        assert sha(root / directory / f) == h
write(
    root / "preflight.json",
    dict(
        passed=True,
        rows=rows,
        changed_batch_checks=checks,
        control_full_state_replay=state,
        final_batch_audit=late,
        source_and_teacher_previews_inherited=True,
    ),
)
print(
    dict(
        preflight=True,
        control_full_state=state,
        max_neutral_difference=max(r["neutral_weight_parity"]["max_abs_difference"] for r in rows),
    ),
    flush=True,
)
