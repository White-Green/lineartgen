"""Audit identical initialization, context gradients and exact FT batch reuse."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from cue_capacity import make_arm
from flax import nnx
from mixed_context_model import infer, load_field
from mixed_context_study import (
    METHODS,
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
from opening_prior_objective import opening_gradient
from plausible_study import exact
from scale_style_balance_study import update
from subpixel_field_model import field_forward

from lineartgen.training import checkpoints, lineart_state

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
root = p.parse_args().root.resolve()
check(root)
assert jax.devices()[0].platform != "cpu"
data, arms = load_data(root), arms_from_root(root)
reference, cfg = load_field(REUSED / "initials/frequent")
assert all(cfg == arm["config"] for arm in arms.values())
control = make_arm(cfg, reference)
control["gradient"] = opening_gradient()
recorded = [json.loads(x) for x in (REUSED / "training/sampling.jsonl").read_text().splitlines()]
assert METHODS == ("mixed",)
forward = nnx.jit(field_forward)
rows = []
for step in range(6):
    b = batch(data, step, "mixed")
    source_step = frequency_plan()["indices"][step]
    exact(b, source_batch(data, source_step, "expanded"))
    values = {}
    for method in METHODS:
        initial, _ = load_field(root / f"initials/{method}")
        parent, _ = load_field(SOURCE / "snapshots/step-6000/expanded/raw")
        fields = replay_state(forward(initial, b[0]), forward(parent, b[0]))
        pixels = replay_state(infer(initial, b[0]), infer(parent, b[0]))
        parts, grad = arms[method]["gradient"](arms[method]["model"], *b)
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves((parts, grad)))
        first = float(np.linalg.norm(np.asarray(grad.context_block.first.kernel)))
        second = float(np.linalg.norm(np.asarray(grad.context_block.second.kernel)))
        assert second > 0
        if step == 0:
            assert first == 0
        values[method] = dict(
            initial_fields=fields,
            initial_pixels=pixels,
            first_gradient_norm=first,
            second_gradient_norm=second,
            metrics=update(arms[method], *b),
        )
    control_stats = replay_state(update(control, *b), recorded[step]["metrics"]["frequent"])
    rows.append(
        dict(
            step=step,
            source_step=source_step,
            depth=step + 1,
            inputs_exact=True,
            methods=values,
            unchanged_base_metrics=control_stats,
        )
    )
for method in METHODS:
    assert max(row["methods"][method]["first_gradient_norm"] for row in rows[1:]) > 0
restored_model, _ = load_field(REUSED / "initials/frequent")
restored = make_arm(cfg, restored_model)
restored["gradient"] = opening_gradient()
with checkpoints.manager(REUSED / "training/checkpoints", keep=8) as saved:
    assert lineart_state.restore(saved, {"frequent": restored}, 6) == 6
state = replay_state(
    lineart_state.payload({"frequent": control}, 6)["arms"]["frequent"],
    lineart_state.payload({"frequent": restored}, 6)["arms"]["frequent"],
)
batch(data, 5999, "mixed")
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
        base_FT_full_state=state,
        final_batch_audit=late,
        source_and_teacher_previews_inherited=True,
    ),
)
print(dict(preflight=True, base_FT_full_state=state), flush=True)
