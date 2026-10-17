"""Audit all-depth initialization and exact GM and own24-update prefix replay."""

import argparse
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import write
from consolidated_geometry_study import replay_state
from cue_capacity import make_arm
from flax import nnx
from opening_prior_objective import opening_gradient
from plausible_study import exact
from pooled_context_model import infer, load_field
from pooled_procedural_study import GM, SAVED, arms_from_root, batch, check, load_data
from scale_style_balance_study import update
from subpixel_field_model import field_forward

from lineartgen.training import checkpoints, lineart_state

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
root = parser.parse_args().root.resolve()
check(root)
assert jax.devices()[0].platform != "cpu"
data, arms = load_data(), arms_from_root(root)
parent, cfg = load_field(GM / "initials/procedural")
assert cfg == arms["pooled"]["config"]
control = make_arm(cfg, parent)
control["gradient"] = opening_gradient()
forward = nnx.jit(field_forward)
initial, _ = load_field(root / "initials/pooled")
reference, _ = load_field(GM / "initials/procedural")
rows, own = [], []
for step in range(24):
    b = batch(data, step)
    if step < 6:
        fields = replay_state(forward(initial, b[0]), forward(reference, b[0]))
        pixels = replay_state(infer(initial, b[0]), infer(reference, b[0]))
        parts, grad = arms["pooled"]["gradient"](arms["pooled"]["model"], *b)
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves((parts, grad)))
        pooled_norm = float(np.linalg.norm(np.asarray(grad.pooled_stem.kernel)))
        assert pooled_norm > 0
        rows.append(
            dict(depth=step + 1, fields=fields, pixels=pixels, pooled_gradient_norm=pooled_norm)
        )
    exact(update(control, *b), data["gm_rows"][step]["metrics"]["procedural"])
    own.append(update(arms["pooled"], *b))
restored_model, cc = load_field(GM / "initials/procedural")
restored = make_arm(cc, restored_model)
restored["gradient"] = opening_gradient()
with checkpoints.manager(GM / "training/checkpoints", keep=len(SAVED)) as manager:
    assert lineart_state.restore(manager, dict(procedural=restored), 24) == 24
exact(
    lineart_state.payload(dict(procedural=control), 24),
    lineart_state.payload(dict(procedural=restored), 24),
)
replay = arms_from_root(root)
for step in range(24):
    exact(update(replay["pooled"], *batch(data, step)), own[step])
exact(lineart_state.payload(replay, 24), lineart_state.payload(arms, 24))
batch(data, 5999)
write(
    root / "preflight.json",
    dict(
        passed=True,
        rows=rows,
        control24_metrics_exact=True,
        control24_full_state_exact=True,
        own24_metrics_exact=True,
        own24_full_state_exact=True,
        last_batch_exact=True,
    ),
)
print(dict(preflight=True, control24_exact=True, own24_exact=True, initials=rows), flush=True)
