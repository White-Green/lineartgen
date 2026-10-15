"""Verify actual six-depth control gradients and unchanged training data."""

import argparse
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from context_dropout_objective import dropout_gradient
from context_dropout_study import PARENT, batch, load_data, replay_state
from merge_field_objective import merged_gradient
from plausible_study import exact
from subpixel_field_model import load_field

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
root = parser.parse_args().root.resolve()
assert jax.devices()[0].platform != "cpu"
data = load_data(root)
model, _ = load_field(root / "reference")
old, new = merged_gradient(), dropout_gradient()
rows = []
for step in range(6):
    a = batch(data, step, "control")
    b = batch(data, step, "dropout")
    exact(a[0], b[0])
    exact(a[1][:3], b[1][:3])
    exact(a[2], b[2])
    expected = old(model, a[0], a[1][:3], a[2])
    actual = new(model, *a)
    audit = replay_state(actual, expected)
    if step == 0:
        exact(actual, new(model, *b))
    rows.append(
        dict(
            depth=step + 1,
            all_keep_gradient=audit,
            dropped_edges=int(np.sum(np.asarray(b[1][3])[1:] == 0)),
        )
    )
inherited = read(PARENT / "test-results.json")
assert inherited["passed"]
for name in (
    "joint_geometry_study",
    "merge_field_objective",
    "regional_field_objective",
    "group_field_model",
    "subpixel_field_model",
):
    file = Path("scripts") / (name + ".py")
    assert sha(file) == sha(PARENT.parent.parent / file)
write(
    root / "preflight.json",
    dict(
        passed=True,
        rows=rows,
        inputs_targets_equal=True,
        depth1_gradient_exact=True,
        inherited_tests=inherited,
    ),
)
