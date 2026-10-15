"""Finalize EK after reviewing every predeclared native comparison page."""

import argparse
import json
from pathlib import Path

import numpy as np
from campaign_protocol import read, sha, write
from geometry_coverage_study import METHODS, PARENT, SAVED, STEPS, check, select

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
root = parser.parse_args().root.resolve()
check(root)
rows = [json.loads(s) for s in (root / "training/sampling.jsonl").read_text().splitlines()]
assert [r["step"] for r in rows] == list(range(1, STEPS + 1))
assert [dict(step=r["step"] - 1, **{k: r[k] for k in select(0)}) for r in rows] == read(
    root / "sampling-plan.json"
)
assert all(len(set(r["input_sha256"].values())) == 1 for r in rows if not any(r["compose"]))
assert read(root / "prefix-audit.json")["passed"]
audit = read(root / "snapshot-audit.json")
assert audit["raw_ema_exact"] and audit["exports"] == 16
for f, h in audit["bindings"].items():
    assert sha(Path(f)) == h
for step in SAVED[1:]:
    assert read(root / f"training/restore-{step}.json")["full_state_exact"]
tests = read(root / "test-results.json")
assert tests["passed"]
for f, h in tests["logs"].items():
    assert sha(root / f) == h
for until in (3000, STEPS):
    assert read(root / f"driver-train-{until}.json")["returncode"] == 0
notes = read(root / "diagnostics/review-observations.json")
assert len(notes) == 28
for group, folder, count in [
    ("evaluation", "evaluation/native-review", 12),
    ("stress", "stress/native-review", 10),
    ("scene512", "scene512", 6),
]:
    manifest = read(root / folder / "manifest.json")["records"]
    assert len(manifest) == count
    files = {r["file"]: r["sha256"] for r in manifest}
    observations = {r["file"]: notes[folder + "/" + r["file"]] for r in manifest}
    for r in manifest:
        assert sha(root / folder / r["file"]) == r["sha256"]
    write(
        root / group / "visual-review.json",
        dict(
            reviewed=True,
            native_scale=True,
            thresholding=False,
            files=files,
            observations=observations,
        ),
    )
for folder in ["evaluation", "stress"]:
    x = read(root / folder / "metrics.json")
    assert len(x["rows"]) == 4608 and x["coverage_exact"] and x["protected_exact"]
    if folder == "evaluation":
        binds = read(root / folder / "prediction-binding.json")
    else:
        binds = read(root / folder / "binding.json")["predictions"]
    for f, h in binds.items():
        assert sha(root / folder / f) == h
scene = read(root / "scene512/metrics.json")
assert scene["model_outputs"] == 192 and scene["protected_exact"]
for f, h in scene["bindings"].items():
    assert sha(Path(f)) == h
pairs = {}
reuse = {}
for group in ("evaluation", "stress"):
    pairs[group] = 0
    reuse[group] = 0
    for p in sorted((root / group / "predictions").glob("*.npz")):
        with np.load(p) as a, np.load(PARENT / group / "predictions" / p.name) as old:
            np.testing.assert_array_equal(a["initial"], old["group"])
            reuse[group] += 1
            if p.name.endswith("-0.npz"):
                with np.load(p.with_name(p.name[:-5] + "1.npz")) as b:
                    for m in ("initial",) + METHODS:
                        np.testing.assert_array_equal(a[m], b[m])
                        pairs[group] += 1
pairs["scene512"] = 0
for p in (root / "scene512").glob("predictions-*-0.npz"):
    with np.load(p) as a, np.load(p.with_name(p.name[:-5] + "1.npz")) as b:
        for k in a.files:
            np.testing.assert_array_equal(a[k], b[k])
            pairs["scene512"] += 1
quantization = {
    m: dict(
        mean=float(np.mean([r["quantization_output_rms"][m] for r in scene["rows"]])),
        maximum=float(max(r["quantization_output_rms"][m] for r in scene["rows"])),
    )
    for m in METHODS
}
changes = {}
for m in METHODS:
    with (
        np.load(root / f"initials/{m}/weights.npz") as a,
        np.load(root / f"snapshots/step-{STEPS}/{m}/raw/weights.npz") as b,
    ):
        delta = np.concatenate([(b[k] - a[k]).ravel() for k in a.files])
        assert np.any(delta)
        changes[m] = dict(rms=float(np.sqrt(np.mean(delta**2))), max_abs=float(abs(delta).max()))
depth_counts = np.zeros((512, 6), np.int32)
orientation_counts = np.zeros((512, 8), np.int32)
mask_counts = np.zeros((512, 3), np.int32)
for row in rows:
    for index, code, mode in zip(row["indices"], row["transforms"], row["modes"], strict=True):
        depth_counts[index, row["depth"] - 1] += 1
        orientation_counts[index, code] += 1
        mask_counts[index, mode] += 1
assert depth_counts.sum() == orientation_counts.sum() == mask_counts.sum() == STEPS * 4
write(
    root / "diagnostics/integrity.json",
    dict(
        gaussian_pairs_exact=pairs,
        reused_initial_predictions_exact=reuse,
        all_training_inputs_identical=False,
        actual_presented_input_differences=sum(
            len(set(r["input_sha256"].values())) > 1 for r in rows
        ),
        geometry_depth_exposures=[int(depth_counts.min()), int(depth_counts.max())],
        geometry_orientation_exposures=[
            int(orientation_counts.min()),
            int(orientation_counts.max()),
        ],
        geometry_mask_exposures=[int(mask_counts.min()), int(mask_counts.max())],
        parameter_changes=changes,
        quantization_output_rms=quantization,
        last600_update_seconds={
            m: float(np.mean([r["seconds"][m] for r in rows[-600:]])) for m in METHODS
        },
    ),
)
evidence = {
    str(p.relative_to(root)): sha(p)
    for p in [
        root / "source.json",
        root / "plan.json",
        root / "test-results.json",
        root / "prefix-audit.json",
        root / "snapshot-audit.json",
        root / "training/sampling.jsonl",
        root / "diagnostics/integrity.json",
        root / "diagnostics/review-observations.json",
        root / "diagnostics/decision.json",
        root / "evaluation/metrics.json",
        root / "stress/metrics.json",
        root / "scene512/metrics.json",
        root / "evaluation/visual-review.json",
        root / "stress/visual-review.json",
        root / "scene512/visual-review.json",
        root / "cache-audit.json",
        root / "initial-audit.json",
    ]
}
assert read(root / "initial-audit.json")["six_depth_functions_exact"]
assert sha(root / "fields.npy") == sha(PARENT / "fields.npy")
assert sha(root / "directions.npy") == sha(PARENT / "directions.npy")
write(
    root / "completion.json",
    dict(
        stage="complete_geometry_coverage",
        steps_per_arm=STEPS,
        model_outputs=9408,
        native_images_reviewed=28,
        adoption=False,
        original_visual_gate_passed=read(root / "diagnostics/decision.json")["visual_gate_passed"],
        fresh_confirmation_evaluated=False,
        evidence=evidence,
    ),
)
write(root / "workflow-status.json", dict(stage="complete_pending_next_decision", step=STEPS))
print(
    dict(
        completed=True,
        outputs=9408,
        native_views=28,
        adoption=False,
        integrity=read(root / "diagnostics/integrity.json"),
    ),
    flush=True,
)
