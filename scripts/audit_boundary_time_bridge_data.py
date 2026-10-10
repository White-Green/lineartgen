"""Verify only time sampling changes before AX learning."""

import inspect
from pathlib import Path

import boundary_time_bridge_toy as new
import jax
import numpy as np
import transport_bridge_toy as old
from campaign_protocol import read, sha, write
from flax import nnx
from test_boundary_time_bridge_toy import audit_batch

from lineartgen.models.serialization import load_inference

root = Path("runs/boundary-time-bridge-toy-20261010").resolve()
plan = read(root / "plan.json")
parent = Path(plan["control_parent"])
functions = (
    "targets",
    "config",
    "initialize",
    "path_inputs",
    "recursive_velocity",
    "endpoint",
    "velocity",
    "velocity_inputs",
    "objective",
    "make_arms",
    "sample",
    "measure",
)
for name in functions:
    assert inspect.getsource(getattr(old, name)) == inspect.getsource(getattr(new, name)), name
assert old.SEED == new.SEED and old.EVAL_SEED == new.EVAL_SEED and old.STEPS == new.STEPS
model, c = load_inference(root / "initial")
control, cc = load_inference(parent / "initial")
assert c == cc
jax.tree.map(
    np.testing.assert_array_equal, nnx.state(model, nnx.Param), nnx.state(control, nnx.Param)
)
rows = [
    audit_batch(int(step), depth)
    for depth in (1, 2, 3)
    for step in np.linspace(0, 5999, 64, dtype=int)
]
assert len(rows) == 192
write(
    root / "data-audit.json",
    dict(
        only_times_changed=True,
        comparisons=len(rows),
        rows=rows,
        source_noise_levels_targets_labels_exact=True,
        condition_specific_label_multisets_exact=True,
        common_initial_parameters_exact=True,
        unchanged_function_source=list(functions),
        audit_script_sha256=sha(Path(__file__)),
        test_script_sha256=sha(Path("tests/test_boundary_time_bridge_toy.py")),
        parent_source_sha256=sha(parent / "source.json"),
    ),
)
old_times = np.array([r["old_time"] for r in rows]).ravel()
new_times = np.array([r["new_time"] for r in rows]).ravel()
print(
    {
        "batches": len(rows),
        "exact_zero_count": int(np.sum(new_times == 0)),
        "old_early_nonzero": int(np.sum((old_times > 0) & (old_times < 0.01))),
        "new_early_nonzero": int(np.sum((new_times > 0) & (new_times < 0.01))),
        "old_late": int(np.sum(old_times > 0.99)),
        "new_late": int(np.sum(new_times > 0.99)),
    }
)
