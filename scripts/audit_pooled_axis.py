"""Check all-six-depth function, shared families, GT24 and own36 full-state replay."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import write
from consolidated_geometry_study import replay_state
from cue_capacity import make_arm
from flax import nnx
from opening_prior_objective import opening_gradient
from plausible_study import exact
from pooled_axis_batches import gp_batch
from pooled_axis_model import infer, load_field
from pooled_axis_study import GT, METHODS, arms_from_root, batch, check, load_data
from scale_style_balance_study import update
from subpixel_field_data import candidate_fields
from subpixel_field_model import coverage, field_forward

from lineartgen.training import checkpoints, lineart_state

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
root = parser.parse_args().root.resolve()
check(root)
assert jax.devices()[0].platform != "cpu"
data, arms = load_data(), arms_from_root(root)
base, cfg = load_field(GT / "initials/no_opening")
forward = nnx.jit(field_forward)
rows = []
for step in (0, 1, 2, 3, 4, 5, 18, 19, 20, 21, 22, 23, 30, 31, 32, 33, 34, 35):
    b = batch(data, step)
    p, c, d = b
    allowed = np.asarray(c[6])
    partial = ~np.all(np.asarray(p.noise_levels[0]) > 0, axis=(1, 2, 3))
    assert not allowed[partial].any()
    values = {}
    for name in METHODS:
        model = arms[name]["model"]
        ferr = replay_state(forward(model, p), forward(base, p))
        perr = replay_state(infer(model, p), infer(base, p))
        parts, grad = arms[name]["gradient"](model, *b)
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves((parts, grad)))
        gpool = float(np.linalg.norm(np.asarray(grad.pooled_stem.kernel)))
        gres = float(np.linalg.norm(np.asarray(grad.head.residual.kernel)))
        assert gpool > 0 and gres > 0
        if not allowed.any():
            old, _ = opening_gradient(weight=0)(model, p, c[:4], d)
            np.testing.assert_allclose(parts["loss"], old["loss"], atol=5e-7, rtol=2e-5)
        values[name] = dict(
            initial_fields=ferr,
            initial_pixels=perr,
            pooled_gradient_norm=gpool,
            residual_gradient_norm=gres,
        )
    if data["last_audit"]["family"] == "ambiguous":
        for field, normal in ((c[0], d), (c[4], c[5])):
            alpha = (
                1 - np.asarray(coverage(candidate_fields(field, normal, len(p.guides))[0][:, 4]))
            ) / 2
            active = np.asarray(p.noise_levels[0]) > 0
            assert np.all((alpha * active).sum(axis=(1, 2, 3))[allowed] > 0)
    rows.append(
        dict(
            step=step, family=data["last_audit"]["family"], allowed=allowed.tolist(), methods=values
        )
    )
# Replay GT with its original architecture and batches, separately from new candidates.
control = make_arm(cfg, base)
control["gradient"] = opening_gradient(weight=0)
gtrows = [json.loads(s) for s in (GT / "training/sampling.jsonl").read_text().splitlines()]
for step in range(24):
    exact(update(control, *gp_batch(data, step)), gtrows[step]["metrics"]["no_opening"])
restored_model, _ = load_field(GT / "initials/no_opening")
restored = dict(no_opening=make_arm(cfg, restored_model))
restored["no_opening"]["gradient"] = opening_gradient(weight=0)
with checkpoints.manager(GT / "training/checkpoints", keep=9) as manager:
    assert lineart_state.restore(manager, restored, 24) == 24
exact(lineart_state.payload(dict(no_opening=control), 24), lineart_state.payload(restored, 24))
metrics = []
for step in range(36):
    b = batch(data, step)
    metrics.append({name: update(arms[name], *b) for name in METHODS})
repeat = arms_from_root(root)
for step in range(36):
    b = batch(data, step)
    for name in METHODS:
        exact(update(repeat[name], *b), metrics[step][name])
exact(lineart_state.payload(repeat, 36), lineart_state.payload(arms, 36))
for step in (5994, 5999):
    batch(data, step)
write(
    root / "preflight.json",
    dict(
        passed=True,
        rows=rows,
        control24_metrics_exact=True,
        control24_full_state_exact=True,
        own36_metrics_exact=True,
        own36_full_state_exact=True,
        last_batches_checked=True,
    ),
)
print(dict(preflight=True, control24_exact=True, own36_exact=True), flush=True)
