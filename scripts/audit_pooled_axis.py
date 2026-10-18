"""HJ: verify restoration, unchanged inference and a continuous optimizer schedule."""

import argparse
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from flax import nnx
from learned_coarse_axis_model import infer, load_field
from plausible_study import exact
from pooled_axis_study import HI, START, STEPS, arms_from_root, batch, check, load_data

from lineartgen.training import checkpoints, lineart_state

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
root = p.parse_args().root.resolve()
check(root)
assert jax.devices()[0].platform != "cpu"
data, arms = load_data(), arms_from_root(root)
with checkpoints.manager(root / "training/checkpoints", keep=15) as manager:
    assert lineart_state.restore(manager, arms, START) == START
    restored = arms_from_root(root)
    assert lineart_state.restore(manager, restored, START) == START
    exact(lineart_state.payload(arms, START), lineart_state.payload(restored, START))
base, _ = load_field(HI / "snapshots/step-6000/consistent/raw")
ema, _ = load_field(HI / "snapshots/step-6000/consistent/ema")
model = arms["consistent"]["model"]
exact(nnx.state(base, nnx.Param), nnx.state(model, nnx.Param))
exact(nnx.state(ema, nnx.Param), arms["consistent"]["ema"])
assert int(arms["consistent"]["optimizer"].step) == START
schedule = arms["consistent"]["schedule"]
lrs = {step: float(schedule(step)) for step in (5999, 6000, 6001, 12000, 17999, 18000)}
np.testing.assert_allclose(list(lrs.values()), 2e-5, atol=1e-10, rtol=0)
rows = []
for index in (0, 1, 2, 3, 4, 5, 18, 19, 20, 21, 22, 23, 30, 31, 32, 33, 34, 35):
    step = START + index
    b = batch(data, step)
    np.testing.assert_array_equal(infer(base, b[0]), infer(model, b[0]))
    parts, grad = arms["consistent"]["gradient"](model, *b)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves((parts, grad)))
    rows.append(dict(step=step, family=data["last_audit"]["family"], initial_pixels_exact=True))
for step in (11994, 11999, 12000, 17994, 17999):
    batch(data, step)
reuse = read(root / "teacher-reuse.json")
for f, h in reuse["files"].items():
    assert sha(Path(f)) == h and sha(root / "support-preview" / Path(f).name) == h
write(
    root / "preflight.json",
    dict(
        passed=True,
        rows=rows,
        full_state_restore_exact=True,
        raw_ema_exact=True,
        optimizer_step=START,
        learning_rates=lrs,
        ending_step=STEPS,
        own36_exact_required=True,
        repeated_input_boundaries_checked=True,
    ),
)
print(dict(preflight=True, optimizer_step=START, learning_rates=lrs), flush=True)
