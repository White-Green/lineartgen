"""Audit original and procedural batches, six depths and an intervention-aware replay."""

import argparse
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from flax import nnx
from mixed_context_model import infer, load_field
from plausible_study import exact
from procedural_shape_study import (
    GH,
    GL,
    METHODS,
    arms_from_root,
    batch,
    check,
    load_data,
    old_batch,
    replay_state,
)
from scale_style_balance_study import update
from subpixel_field_data import candidate_fields
from subpixel_field_model import coverage, field_forward

from lineartgen.training import lineart_state

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
root = p.parse_args().root.resolve()
check(root)
assert read(GH / "completion.json")["stage"] == "complete_residual_axis"
assert jax.devices()[0].platform != "cpu"
data, arms = load_data(root), arms_from_root(root)
base, _ = load_field(root / "reference")
forward = nnx.jit(field_forward)
initial_rows = []
for step in range(6):
    a = batch(data, step, "control")
    exact(a, batch(data, step, "procedural"))
    exact(a, old_batch(data, step, "choice"))
    values = {}
    for name in METHODS:
        initial = arms[name]["model"]
        values[name] = dict(
            fields=replay_state(forward(initial, a[0]), forward(base, a[0])),
            pixels=replay_state(infer(initial, a[0]), infer(base, a[0])),
        )
    initial_rows.append(dict(depth=step + 1, values=values))
procedural_rows = []
for step in range(18, 24):
    a, b = batch(data, step, "control"), batch(data, step, "procedural")
    exact(a, old_batch(data, step, "choice"))
    exact(a[0].noise_levels, b[0].noise_levels)
    exact(a[1][3], b[1][3])
    depth = len(b[0].guides)
    assert depth == step % 6 + 1 and not np.asarray(b[1][2]).any()
    exact(b[1][0], b[1][1])
    with np.load(GL / f"preview-depth-{depth}.npz") as z:
        np.testing.assert_array_equal(b[0].guides[0], z["source"])
        np.testing.assert_array_equal(b[1][0], z["base"])
        np.testing.assert_array_equal(b[2], z["direction"])
        field = candidate_fields(b[1][0], b[2], depth)[0][:, 4]
        rendered = np.asarray(coverage(field))
        err = float(abs(rendered - z["clean"]).max())
        assert err <= 1e-6
    parts, grads = arms["procedural"]["gradient"](arms["procedural"]["model"], *b)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves((parts, grads)))
    norm = float(np.sqrt(sum(float(np.sum(np.asarray(x) ** 2)) for x in jax.tree.leaves(grads))))
    assert norm > 0
    procedural_rows.append(
        dict(
            step=step,
            depth=depth,
            preview_clean_max_abs=err,
            source_base_directions_exact=True,
            masks_dropout_exact=True,
            gradient_norm=norm,
        )
    )
rows = []
for step in range(24):
    rows.append({m: update(arms[m], *batch(data, step, m)) for m in METHODS})
    assert all(np.isfinite(v) for stats in rows[-1].values() for v in stats.values())
repeated = arms_from_root(root)
for step in range(24):
    for m in METHODS:
        exact(update(repeated[m], *batch(data, step, m)), rows[step][m])
exact(lineart_state.payload(arms, 24), lineart_state.payload(repeated, 24))
write(
    root / "preflight.json",
    dict(
        passed=True,
        initial=initial_rows,
        procedural=procedural_rows,
        first24_metrics_full_state_exact=True,
        intervention_updates_in_prefix=6,
        prerequisite_bindings={
            str((GH / f).resolve()): sha(GH / f)
            for f in ("completion.json", "diagnostics/decision.json")
        },
    ),
)
print(dict(preflight=True, original_depths=6, procedural_depths=6, replay_updates=24), flush=True)
