"""Verify condition-preserving optimal coupling before AW learning."""

import inspect
from pathlib import Path

import jax
import numpy as np
import transport_bridge_toy as new
import velocity_bridge_toy as old
from campaign_protocol import read, sha, write
from flax import nnx
from test_transport_bridge_toy import audit_batch

from lineartgen.models.serialization import load_inference

root = Path("runs/transport-bridge-toy-20261010").resolve()
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
        conditional_optimal_assignment_verified=True,
        comparisons=len(rows),
        rows=rows,
        source_noise_levels_progress_times_exact=True,
        condition_specific_label_multisets_exact=True,
        common_initial_parameters_exact=True,
        unchanged_function_source=list(functions),
        audit_script_sha256=sha(Path(__file__)),
        test_script_sha256=sha(Path("tests/test_transport_bridge_toy.py")),
        parent_source_sha256=sha(parent / "source.json"),
    ),
)
print(
    {
        "batches": len(rows),
        "changed_labels": sum(r["changed_labels"] for r in rows),
        "independent_cost": sum(r["independent_cost"] for r in rows),
        "transport_cost": sum(r["transport_cost"] for r in rows),
    }
)
