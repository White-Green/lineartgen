"""Prove evaluator repair leaves the six startup updates unchanged."""

import argparse
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import sha, write
from train_paper_band_long_run import check, make_arms

from lineartgen.training import checkpoints, lineart_state

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--root", type=Path, required=True)
r = p.parse_args().root.resolve()
plan = check(r)
old = Path(plan["replay_source"])
a, b = make_arms(r), make_arms(r)
for directory, arms in [(r, a), (old, b)]:
    with checkpoints.manager(directory / "milestone-checkpoints", keep=4) as saved:
        assert lineart_state.restore(saved, arms, 4006) == 4006
jax.tree.map(
    np.testing.assert_array_equal, lineart_state.payload(a, 4006), lineart_state.payload(b, 4006)
)
assert (r / "training/sampling.jsonl").read_text() == (old / "training/sampling.jsonl").read_text()
with (
    np.load(r / "evaluation/step-4000/predictions.npz") as a,
    np.load(old / "evaluation/step-4000/predictions.npz") as b,
):
    assert set(a.files) == set(b.files)
    for key in a.files:
        np.testing.assert_array_equal(a[key], b[key])
write(
    r / "startup-replay-audit.json",
    dict(
        model_optimizer_ema_exact=True,
        all_six_sampling_records_exact=True,
        all_initial_predictions_exact=True,
        step=4006,
        old_source_sha256=sha(old / "source.json"),
        old_checkpoint_files={
            str(p): sha(p) for p in (old / "milestone-checkpoints/4006").rglob("*") if p.is_file()
        },
    ),
)
