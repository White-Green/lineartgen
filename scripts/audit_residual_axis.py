"""Audit initial function, identical payloads, mask guard and finite gradients."""

import argparse
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, write
from flax import nnx
from opening_prior_objective import opening_gradient
from plausible_study import exact
from residual_axis_model import infer, load_field
from residual_axis_study import (
    GG,
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
assert jax.devices()[0].platform != "cpu"
data, arms = load_data(root), arms_from_root(root)
base, _ = load_field(root / "reference")
forward = nnx.jit(field_forward)
rows = []
for step in range(6):
    b = batch(data, step, "mean")
    exact(b, batch(data, step, "choice"))
    previous = old_batch(data, step, "choice")
    exact((b[0], b[1][:4], b[2]), previous)
    assert not np.asarray(b[1][6]).any()
    values = {}
    for name in METHODS:
        initial, _ = load_field(root / f"initials/{name}")
        fields = replay_state(forward(initial, b[0]), forward(base, b[0]))
        pixels = replay_state(infer(initial, b[0]), infer(base, b[0]))
        parts, grads = arms[name]["gradient"](arms[name]["model"], *b)
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves((parts, grads)))
        g = float(np.linalg.norm(np.asarray(grads.head.residual.kernel)))
        assert g > 0
        legacy, old_grad = opening_gradient()(arms[name]["model"], *previous)
        unchanged = replay_state(
            {k: v for k, v in parts.items() if k != "axis_eligible_fraction"}, legacy
        )
        gradient_equal = replay_state(grads, old_grad)
        values[name] = dict(
            initial_fields=fields,
            initial_pixels=pixels,
            residual_gradient_norm=g,
            disabled_objective=unchanged,
            disabled_gradient=gradient_equal,
            metrics=update(arms[name], *b),
        )
    rows.append(dict(step=step, depth=step + 1, original_batch_exact=True, methods=values))
repeated = arms_from_root(root)
for step in range(6):
    b = batch(data, step, "mean")
    for name in METHODS:
        exact(update(repeated[name], *b), rows[step]["methods"][name]["metrics"])
exact(lineart_state.payload(arms, 6), lineart_state.payload(repeated, 6))
eligible = []
for selected in read(GG / "eligibility.json")["selected"]:
    step = selected["step"]
    b = batch(data, step, "mean")
    allowed = np.asarray(b[1][6])
    assert allowed[selected["item"]]
    assert np.all(np.asarray(b[0].noise_levels[0])[allowed] > 0)
    depth, item = selected["depth"], selected["item"]
    teacher_errors = {}
    with np.load(GG / f"depth-{depth}.npz") as preview:
        for label, fields, directions in (
            ("original", b[1][0], b[2]),
            ("alternative", b[1][4], b[1][5]),
        ):
            field = candidate_fields(fields, directions, depth)[0][item, 4]
            old_field = preview[label + "_field"]
            field_error = float(np.max(abs(np.clip(field, -8, 8) - np.clip(old_field, -8, 8))))
            pixels = np.asarray(coverage(field[None]))[0]
            pixel_error = float(np.max(abs(pixels - preview[label])))
            # Independent CPU/GPU construction is stored as float16 fields.
            assert field_error <= 1 / 256 and pixel_error <= 1e-3
            teacher_errors[label] = dict(field_max_abs=field_error, pixel_max_abs=pixel_error)
    stats = {}
    for name in METHODS:
        parts, grads = arms[name]["gradient"](arms[name]["model"], *b)
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves((parts, grads)))
        assert float(parts["axis_eligible_fraction"]) == allowed.mean()
        stats[name] = {k: float(v) for k, v in parts.items()}
    eligible.append(
        dict(
            step=step,
            depth=depth,
            allowed=allowed.tolist(),
            metrics=stats,
            preview_errors=teacher_errors,
        )
    )
write(
    root / "preflight.json",
    dict(
        passed=True,
        rows=rows,
        eligible_examples=eligible,
        six_update_replay_and_full_state_exact=True,
        initial_array_identity_verified_at_prepare=True,
    ),
)
print(dict(preflight=True, eligible_batches=len(eligible)), flush=True)
