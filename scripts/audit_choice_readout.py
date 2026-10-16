"""Audit identical initialization, context gradients and exact FT batch reuse."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from choice_readout_model import infer, load_field
from choice_readout_study import (
    LOCALCONTROL,
    METHODS,
    arms_from_root,
    batch,
    check,
    frequency_plan,
    load_data,
    replay_state,
    source_batch,
)
from cue_capacity import make_arm
from flax import nnx
from group_field_model import GroupHead
from opening_prior_objective import opening_gradient
from plausible_study import exact
from scale_style_balance_study import update
from subpixel_field_model import field_forward

from lineartgen.training import checkpoints, lineart_state


class ManualFirst(GroupHead):
    def __call__(self, x):
        b, h, w, c = x.shape
        # Keep the full projection layout before selecting orientation0.
        # Selecting the feature first changes GEMM shape/rounding on ROCm.
        projected = self.linear(x.reshape(b, h, w, 8, c // 8))
        return projected[..., 0, :]


p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
root = p.parse_args().root.resolve()
check(root)
assert jax.devices()[0].platform != "cpu"
data, arms = load_data(root), arms_from_root(root)
reference, cfg = load_field(LOCALCONTROL / "initials/mixed")
assert all(cfg == arm["config"] for arm in arms.values())
control = make_arm(cfg, reference)
control["gradient"] = opening_gradient()
recorded = [
    json.loads(x) for x in (LOCALCONTROL / "training/sampling.jsonl").read_text().splitlines()
]
assert METHODS == ("choice",)
forward = nnx.jit(field_forward)
rows = []
for step in range(6):
    b = batch(data, step, "choice")
    source_step = frequency_plan()["indices"][step]
    exact(b, source_batch(data, source_step, "expanded"))
    values = {}
    for method in METHODS:
        initial, _ = load_field(root / f"initials/{method}")
        parent, _ = load_field(LOCALCONTROL / "initials/mixed")
        mean_pixels = np.asarray(infer(parent, b[0]))
        head = ManualFirst(parent.width, rngs=nnx.Rngs(0))
        nnx.update(head, nnx.state(parent.head, nnx.Param))
        parent.head = head
        fields = replay_state(forward(initial, b[0]), forward(parent, b[0]))
        pixels = replay_state(infer(initial, b[0]), infer(parent, b[0]))
        parts, grad = arms[method]["gradient"](arms[method]["model"], *b)
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves((parts, grad)))
        first = float(np.linalg.norm(np.asarray(grad.context_block.first.kernel)))
        second = float(np.linalg.norm(np.asarray(grad.context_block.second.kernel)))
        head_gradient = float(np.linalg.norm(np.asarray(grad.head.linear.kernel)))
        assert head_gradient > 0
        assert second > 0
        if step == 0:
            assert first == 0
        values[method] = dict(
            initial_fields=fields,
            initial_pixels=pixels,
            first_gradient_norm=first,
            second_gradient_norm=second,
            head_gradient_norm=head_gradient,
            initial_difference_from_mean=float(
                np.max(abs(np.asarray(infer(initial, b[0])) - mean_pixels))
            ),
            metrics=update(arms[method], *b),
        )
    control_stats = replay_state(update(control, *b), recorded[step]["metrics"]["mixed"])
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
restored_model, _ = load_field(LOCALCONTROL / "initials/mixed")
restored = make_arm(cfg, restored_model)
restored["gradient"] = opening_gradient()
with checkpoints.manager(LOCALCONTROL / "training/checkpoints", keep=8) as saved:
    assert lineart_state.restore(saved, {"mixed": restored}, 6) == 6
state = replay_state(
    lineart_state.payload({"mixed": control}, 6)["arms"]["mixed"],
    lineart_state.payload({"mixed": restored}, 6)["arms"]["mixed"],
)
batch(data, 5999, "choice")
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
        base_FW_full_state=state,
        final_batch_audit=late,
        source_and_teacher_previews_inherited=True,
    ),
)
print(dict(preflight=True, base_FW_full_state=state), flush=True)
