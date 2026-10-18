"""HM: full-state handoff, all-depth inputs, and direct HL update parity."""

import argparse
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from flax import nnx
from late_opening_study import arms_from_root as hl_arms
from learned_coarse_axis_model import infer, load_field
from plausible_study import exact
from pooled_axis_study import HL, METHODS, START, STEPS, arms_from_root, batch, check, load_data
from scale_style_balance_study import update

from lineartgen.training import checkpoints, lineart_state

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
root = p.parse_args().root.resolve()
check(root)
assert jax.devices()[0].platform != "cpu"
data, arms = load_data(), arms_from_root(root)
old = hl_arms(HL)
with checkpoints.manager(HL / "training/checkpoints", keep=6) as manager:
    assert lineart_state.restore(manager, old, START) == START
exact(lineart_state.payload(old, START), lineart_state.payload(arms, START))
for name in METHODS:
    base, cfg = load_field(HL / f"snapshots/step-{START}/{name}/raw")
    exact(nnx.state(base, nnx.Param), nnx.state(arms[name]["model"], nnx.Param))
    assert cfg == arms[name]["config"]
    assert int(arms[name]["optimizer"].step) == START
    np.testing.assert_allclose(
        np.asarray(arms[name]["schedule"](jax.numpy.arange(START, STEPS + 1))),
        2e-5,
        atol=1e-10,
        rtol=0,
    )
rows = []
for offset in (0, 1, 2, 3, 4, 5, 18, 19, 20, 21, 22, 23, 30, 31, 32, 33, 34, 35):
    step = START + offset
    b = batch(data, step)
    parts = {}
    for name in METHODS:
        np.testing.assert_array_equal(
            infer(arms[name]["model"], b[0]), infer(old[name]["model"], b[0])
        )
        pp, gg = arms[name]["gradient"](arms[name]["model"], *b)
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves((pp, gg)))
        assert any(np.any(np.asarray(x) != 0) for x in jax.tree.leaves(gg))
        parts[name] = {k: float(v) for k, v in pp.items()}
    rows.append(
        dict(step=step, family=data["last_audit"]["family"], initial_pixels_exact=True, parts=parts)
    )
for step in range(START, START + 6):
    b = batch(data, step)
    for name in METHODS:
        exact(update(old[name], *b), update(arms[name], *b))
exact(lineart_state.payload(old, START + 6), lineart_state.payload(arms, START + 6))
for step in (STEPS - 6, STEPS - 1):
    batch(data, step)
for f, h in read(root / "teacher-reuse.json")["files"].items():
    assert sha(Path(f)) == h and sha(root / "support-preview" / Path(f).name) == h
write(
    root / "preflight.json",
    dict(
        passed=True,
        rows=rows,
        inherited_full_state_exact=True,
        inherited_optimizer_step=START,
        learning_rate_constant=2e-5,
        direct_hl6_full_state_exact=True,
        all_arms_own36_exact_required=True,
        final_batches_checked=True,
    ),
)
print(
    dict(preflight=True, inherited_full_state_exact=True, direct_hl6_full_state_exact=True),
    flush=True,
)
