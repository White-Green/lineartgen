"""GY preflight: unchanged initialization, all families/depths, archived GW36 replay."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import write
from cue_capacity import make_arm
from flax import nnx
from plausible_study import exact
from pooled_axis_model import infer, load_field
from pooled_axis_objective import axis_gradient as winner_gradient
from pooled_axis_study import GW, arms_from_root, batch, check, load_data
from scale_style_balance_study import update
from subpixel_field_data import candidate_fields
from subpixel_field_model import coverage

from lineartgen.training import checkpoints, lineart_state

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
root = p.parse_args().root.resolve()
check(root)
assert jax.devices()[0].platform != "cpu"
data, arms = load_data(), arms_from_root(root)
base, cfg = load_field(GW / "initials/choice")
exact(nnx.state(base, nnx.Param), nnx.state(arms["consistent"]["model"], nnx.Param))
original_gradient = winner_gradient(prior_mode="none")
rows = []
for step in (0, 1, 2, 3, 4, 5, 18, 19, 20, 21, 22, 23, 30, 31, 32, 33, 34, 35):
    b = batch(data, step)
    prepared, c, d = b
    allowed = np.asarray(c[6])
    partial = ~np.all(np.asarray(prepared.noise_levels[0]) > 0, axis=(1, 2, 3))
    assert not allowed[partial].any()
    np.testing.assert_array_equal(
        infer(base, prepared), infer(arms["consistent"]["model"], prepared)
    )
    parts, grad = arms["consistent"]["gradient"](arms["consistent"]["model"], *b)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves((parts, grad)))
    gp = float(np.linalg.norm(np.asarray(grad.pooled_stem.kernel)))
    gr = float(np.linalg.norm(np.asarray(grad.head.residual.kernel)))
    assert gp > 0 and gr > 0
    if not allowed.any():
        old, _ = original_gradient(base, *b)
        for key in old:
            np.testing.assert_allclose(parts[key], old[key], atol=1e-6, rtol=2e-5)
    if data["last_audit"]["family"] == "ambiguous":
        for field, normal in ((c[0], d), (c[4], c[5])):
            alpha = (
                1
                - np.asarray(
                    coverage(candidate_fields(field, normal, len(prepared.guides))[0][:, 4])
                )
            ) / 2
            assert np.all(
                (alpha * (np.asarray(prepared.noise_levels[0]) > 0)).sum(axis=(1, 2, 3))[allowed]
                > 0
            )
    rows.append(
        dict(
            step=step,
            family=data["last_audit"]["family"],
            allowed=allowed.tolist(),
            initial_pixels_exact=True,
            pooled_gradient_norm=gp,
            residual_gradient_norm=gr,
        )
    )
control = make_arm(cfg, base)
control["gradient"] = original_gradient
previous = [json.loads(s) for s in (GW / "training/sampling.jsonl").read_text().splitlines()]
errors = []
for step in range(36):
    actual = update(control, *batch(data, step))
    expected = previous[step]["metrics"]["choice"]
    assert actual.keys() == expected.keys()
    for k in actual:
        np.testing.assert_allclose(actual[k], expected[k], atol=1e-6, rtol=2e-5)
    errors.append(max(abs(actual[k] - expected[k]) for k in actual))
restored = {}
for name in ("mean", "choice"):
    m, cfg = load_field(GW / f"initials/{name}")
    restored[name] = make_arm(cfg, m)
    restored[name]["gradient"] = original_gradient
with checkpoints.manager(GW / "training/checkpoints", keep=9) as manager:
    assert lineart_state.restore(manager, restored, 36) == 36
actual = lineart_state.payload({"choice": control}, 36)
expected = lineart_state.payload({"choice": restored["choice"]}, 36)
state_errors = []
for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind in "biu":
        np.testing.assert_array_equal(a, b)
    else:
        np.testing.assert_allclose(a, b, atol=1e-6, rtol=2e-5)
    state_errors.append(float(np.max(abs(a.astype(float) - b.astype(float)))))
for step in (5994, 5999):
    batch(data, step)
write(
    root / "preflight.json",
    dict(
        passed=True,
        rows=rows,
        control36_metrics_close=True,
        control36_metrics_max_abs=max(errors),
        control36_full_state_close=True,
        control36_state_max_abs=max(state_errors),
        cross_process_tolerance=dict(atol=1e-6, rtol=2e-5),
        own36_exact_required_in_training=True,
        last_batches_checked=True,
    ),
)
print(dict(preflight=True, control36_close=True), flush=True)
