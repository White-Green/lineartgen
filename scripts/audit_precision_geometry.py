"""Verify exact teacher families and storage/noise ordering across all six depths."""

import argparse
from pathlib import Path

import jax
import merge_geometry_study as previous
import numpy as np
from campaign_protocol import read, sha, write
from precision_geometry_study import CONTROL, SEED, batch, load_data
from precision_source_data import quant_choice, quantize_source

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
root = p.parse_args().root.resolve()
data = load_data(root)
for step in range(6):
    a = batch(data, step, "quant")
    b = previous.batch(data, step, "merge")
    jax.tree.map(np.testing.assert_array_equal, a[1:], b[1:])
    jax.tree.map(np.testing.assert_array_equal, a[0].noise_levels, b[0].noise_levels)
    for ta, tb, mask, ga, gb, na, nb in zip(
        a[0].targets,
        b[0].targets,
        a[0].noise_levels,
        a[0].guides,
        b[0].guides,
        a[0].noisy,
        b[0].noisy,
        strict=True,
    ):
        active = np.asarray(mask) > 0
        np.testing.assert_array_equal(np.asarray(ta)[active], np.asarray(tb)[active])
        np.testing.assert_array_equal(np.asarray(ta)[~active], np.asarray(ga)[~active])
        np.testing.assert_allclose(na - ga, nb - gb, atol=1.2e-7, rtol=0)
    unchanged = [i for i in range(4) if not quant_choice(SEED, step, i)]
    if unchanged:
        jax.tree.map(
            lambda x, y, unchanged=unchanged: np.testing.assert_array_equal(
                np.asarray(x)[unchanged], np.asarray(y)[unchanged]
            ),
            a[0],
            b[0],
        )
for i in range(8):
    with np.load(root / f"scene512/input-{i}.npz") as z:
        np.testing.assert_array_equal(quantize_source(z["source"]), z["quantized"])
inherited = read(CONTROL / "test-results.json")
assert inherited["passed"]
for name in [
    "scripts/merge_field_objective.py",
    "scripts/merge_field_data.py",
    "scripts/group_field_model.py",
    "scripts/continuous_width_data.py",
    "scripts/d4_geometry_data.py",
]:
    assert sha(Path(name)) == sha(CONTROL.parent.parent / name)
write(
    root / "preflight.json",
    dict(
        passed=True,
        six_depth_inputs_and_targets_verified=True,
        stored_eight_scenes_exact=True,
        inherited_tests=inherited,
        inherited_preflight_sha256=sha(CONTROL / "preflight.json"),
    ),
)
