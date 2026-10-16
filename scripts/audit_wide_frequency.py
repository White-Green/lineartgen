"""Check the frozen resampling schedule and exact unchanged-objective updates."""

import argparse
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from cue_capacity import make_arm
from opening_prior_objective import opening_gradient
from physical_field_model import infer, load_field
from plausible_study import exact
from scale_style_balance_study import update
from wide_frequency_study import (
    REUSED,
    SOURCE,
    arms_from_root,
    batch,
    check,
    frequency_plan,
    load_data,
    replay_state,
    source_batch,
)

from lineartgen.training import lineart_state

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
root = p.parse_args().root.resolve()
check(root)
assert jax.devices()[0].platform != "cpu"
data, arms = load_data(root), arms_from_root(root)
reference, cfg = load_field(REUSED / "initials/uniform")
assert cfg == arms["frequent"]["config"]
old = make_arm(cfg, reference)
old["gradient"] = opening_gradient()
rows = []
for step in range(6):
    b = batch(data, step, "frequent")
    source_step = frequency_plan()["indices"][step]
    direct = source_batch(data, source_step, "expanded")
    exact(b, direct)
    initial, _ = load_field(root / "initials/frequent")
    parent, _ = load_field(SOURCE / "snapshots/step-6000/expanded/raw")
    exact(infer(initial, b[0]), infer(parent, b[0]))
    parts, grad = arms["frequent"]["gradient"](arms["frequent"]["model"], *b)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves((parts, grad)))
    assert max(float(abs(np.asarray(x)).max()) for x in jax.tree.leaves(grad)) > 0
    actual, expected = update(arms["frequent"], *b), update(old, *direct)
    rows.append(
        dict(
            step=step,
            source_step=source_step,
            depth=step + 1,
            inputs_exact=True,
            initial_exact=True,
            gradients_finite_nonzero=True,
            metrics=replay_state(actual, expected),
            teacher_audit=data["last_audit"].copy(),
        )
    )
state = replay_state(
    lineart_state.payload({"frequent": arms["frequent"]}, 6)["arms"]["frequent"],
    lineart_state.payload({"frequent": old}, 6)["arms"]["frequent"],
)
batch(data, 5999, "frequent")
late = data["last_audit"].copy()
for directory in ("teacher-preview", "source-preview"):
    review = read(root / directory / "visual-review.json")
    assert review["reviewed"]
    for file, expected in review["files"].items():
        assert sha(root / directory / file) == expected
write(
    root / "preflight.json",
    dict(
        passed=True,
        rows=rows,
        unchanged_objective_full_state=state,
        final_batch_audit=late,
        source_and_teacher_previews_inherited=True,
    ),
)
print(dict(preflight=True, full_state=state), flush=True)
