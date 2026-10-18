"""HP: verify HO initialization and unchanged-opening control update parity."""

import argparse
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from coherent_broad_objective import axis_gradient as old_gradient
from flax import nnx
from learned_coarse_axis_model import infer, load_field
from plausible_study import exact
from pooled_axis_study import HO, METHODS, STEPS, arms_from_root, batch, check, load_data
from scale_style_balance_study import update

from lineartgen.training import lineart_state

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
root = p.parse_args().root.resolve()
check(root)
assert jax.devices()[0].platform != "cpu"
data, arms = load_data(), arms_from_root(root)
base, _ = load_field(HO / "snapshots/step-6000/consistent/raw")
for arm in arms.values():
    exact(nnx.state(base, nnx.Param), nnx.state(arm["model"], nnx.Param))
    exact(nnx.state(base, nnx.Param), arm["ema"])
    assert int(arm["optimizer"].step) == 0
    steps = np.array([0, STEPS // 2, STEPS], dtype=np.float32)
    expected_lr = 2e-5 + 0.5 * (2e-4 - 2e-5) * (1 + np.cos(np.pi * steps / STEPS))
    np.testing.assert_allclose(arm["schedule"](steps), expected_lr, atol=1e-10, rtol=1e-6)
rows = []
for step in (0, 1, 2, 3, 4, 5, 18, 19, 20, 21, 22, 23, 30, 31, 32, 33, 34, 35):
    b = batch(data, step)
    expected = np.asarray(infer(base, b[0]))
    parts = {}
    for name in METHODS:
        np.testing.assert_array_equal(infer(arms[name]["model"], b[0]), expected)
        pp, gg = arms[name]["gradient"](arms[name]["model"], *b)
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves((pp, gg)))
        assert any(np.any(np.asarray(x) != 0) for x in jax.tree.leaves(gg))
        parts[name] = {k: float(v) for k, v in pp.items()}
    np.testing.assert_allclose(
        parts["consistent"]["loss"] - parts["winner"]["loss"],
        0.1 * parts["winner"]["support_margin"],
        atol=1e-7,
        rtol=1e-5,
    )
    assert parts["consistent"]["support_margin"] == parts["winner"]["support_margin"]
    rows.append(
        dict(step=step, family=data["last_audit"]["family"], initial_pixels_exact=True, parts=parts)
    )
assert any(r["parts"]["winner"]["support_margin"] > 0 for r in rows)
old = arms_from_root(root)["winner"]
old["gradient"] = old_gradient(opening_weight=0.1)
new = arms_from_root(root)["winner"]
for step in range(6):
    b = batch(data, step)
    a, z = update(old, *b), update(new, *b)
    exact(a, {k: z[k] for k in a})
exact(lineart_state.payload({"control": old}, 6), lineart_state.payload({"control": new}, 6))
for step in (1194, 1199):
    batch(data, step)
for f, h in read(root / "teacher-reuse.json")["files"].items():
    assert sha(Path(f)) == h and sha(root / "support-preview" / Path(f).name) == h
write(
    root / "preflight.json",
    dict(
        passed=True,
        rows=rows,
        initial_parameters_exact=True,
        fresh_optimizer_step=0,
        learning_rate_cosine=dict(peak=2e-4, minimum=2e-5, steps=STEPS),
        old_control6_full_state_exact=True,
        all_arms_own36_exact_required=True,
        final_batches_checked=True,
    ),
)
print(dict(preflight=True, old_control6_full_state_exact=True), flush=True)
