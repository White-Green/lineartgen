"""Bind the matched full-data residual comparison and its actual visual review."""

import argparse
import json
from pathlib import Path

import numpy as np
from campaign_protocol import read, sha, write
from procedural_shape_study import (
    GH,
    GL,
    LOCALCONTROL,
    METHODS,
    REPORT_METHODS,
    SAVED,
    STEPS,
    check,
    select,
    selected,
)

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
root = p.parse_args().root.resolve()
check(root)
rows = [json.loads(x) for x in (root / "training/sampling.jsonl").read_text().splitlines()]
old = [json.loads(x) for x in (LOCALCONTROL / "training/sampling.jsonl").read_text().splitlines()]
assert len(rows) == STEPS
recipes = {(r["step"], r["item"]): r for r in read(GL / "recipes.json")}
changed = []
for step, (a, b) in enumerate(zip(rows, old, strict=True)):
    assert a["step"] == step + 1
    for k, v in select(step).items():
        assert a[k] == v
    assert a["context_keep"] == b["context_keep"]
    assert a["input_sha256"]["control"] == b["input_sha256"]["mixed"]
    assert a["candidate_payload_sha256"]["control"] == b["candidate_payload_sha256"]["mixed"]
    assert a["merge_eligible"]["control"] == b["merge_eligible"]["mixed"]
    assert a["teacher_audit"]["procedural_selected"] == selected(step)
    for m in METHODS:
        assert a["metrics"][m]["merged_fraction"] == sum(a["merge_eligible"][m]) / 4
        assert all(np.isfinite(v) for v in a["metrics"][m].values())
    if selected(step):
        assert a["input_sha256"]["control"] != a["input_sha256"]["procedural"]
        assert (
            a["candidate_payload_sha256"]["control"] != a["candidate_payload_sha256"]["procedural"]
        )
        assert not any(a["merge_eligible"]["procedural"])
        assert a["teacher_audit"]["procedural_records"] == [recipes[step, i] for i in range(4)]
        changed.append(step)
    else:
        assert (
            len(set(a["input_sha256"].values()))
            == len(set(a["candidate_payload_sha256"].values()))
            == 1
        )
assert len(changed) == 1500
assert [sum(s % 6 == d for s in changed) for d in range(6)] == [250] * 6
assert read(root / "preflight.json")["passed"] and read(root / "prefix-audit.json")["passed"]
for f, h in read(root / "preflight.json")["prerequisite_bindings"].items():
    assert sha(Path(f)) == h
audit = read(root / "snapshot-audit.json")
assert audit["raw_ema_exact"] and audit["exports"] == 36
for f, h in audit["bindings"].items():
    assert sha(Path(f)) == h
for step in SAVED[1:]:
    assert read(root / f"training/restore-{step}.json")["full_state_exact"]
phases = [
    "preflight",
    *[f"train-{s}" for s in SAVED[3:]],
    "evaluation",
    "stress",
    "scene512",
    "width-sweep",
    "ambiguity",
    "geometry",
]
for phase in phases:
    assert read(root / f"driver-{phase}.json")["returncode"] == 0
assert read(root / "test-results.json")["passed"]
for folder in ("source-preview", "teacher-preview"):
    review = read(root / folder / "visual-review.json")
    assert review["reviewed"]
    for f, h in review["files"].items():
        assert sha(root / folder / f) == h
assert read(GH / "completion.json")["stage"] == "complete_residual_axis"
assert read(GL / "decision.json")["teacher_preview_passed"]
notes = read(root / "diagnostics/review-observations.json")
assert len(notes) == 34
for group, folder, count in [
    ("evaluation", "evaluation/native-review", 12),
    ("stress", "stress/native-review", 10),
    ("scene512", "scene512", 6),
    ("width-sweep", "width-sweep", 6),
]:
    records = read(root / folder / "manifest.json")["records"]
    assert len(records) == count
    for r in records:
        assert sha(root / folder / r["file"]) == r["sha256"] and notes[folder + "/" + r["file"]]
    write(
        root / group / "visual-review.json",
        dict(
            reviewed=True,
            native_scale=True,
            files={r["file"]: r["sha256"] for r in records},
            observations={r["file"]: notes[folder + "/" + r["file"]] for r in records},
        ),
    )
reference_counts = {}
for group in ("evaluation", "stress"):
    metrics = read(root / group / "metrics.json")
    assert len(metrics["rows"]) == 4608 and metrics["protected_exact"] and metrics["coverage_exact"]
    bindings = (
        read(root / group / "prediction-binding.json")
        if group == "evaluation"
        else read(root / group / "binding.json")["predictions"]
    )
    for f, h in bindings.items():
        assert sha(root / group / f) == h
    count = 0
    for p in (root / group / "predictions").glob("*.npz"):
        with np.load(p) as a, np.load(LOCALCONTROL / group / "predictions" / p.name) as b:
            np.testing.assert_array_equal(a["initial"], b["mixed"])
        count += 1
    reference_counts[group] = count
for group, key, count in [("scene512", "model_outputs", 192), ("width-sweep", "outputs", 108)]:
    m = read(root / group / "metrics.json")
    assert m[key] == count
    for f, h in m["bindings"].items():
        assert sha(Path(f)) == h
assert read(root / "scene512/metrics.json")["protected_exact"]
assert read(root / "width-sweep/metrics.json")["states_unchanged"]
for p in (root / "scene512").glob("predictions-*.npz"):
    with np.load(p) as a, np.load(LOCALCONTROL / "scene512" / p.name) as b:
        for encoding in ("float", "uint8"):
            np.testing.assert_array_equal(a["initial_" + encoding], b["mixed_" + encoding])
for p in (root / "width-sweep").glob("wide-*.npz"):
    with np.load(p) as a, np.load(LOCALCONTROL / "width-sweep" / p.name) as b:
        np.testing.assert_array_equal(a["source"], b["source"])
        np.testing.assert_array_equal(a["initial"], b["mixed"])
for group, outputs, pages in [("ambiguity", 12, 3), ("geometry", 144, 9)]:
    folder = root / "diagnostics" / group
    metrics = read(folder / "metrics.json")
    assert metrics["outputs"] == outputs and metrics["states_unchanged"]
    if group == "geometry":
        assert metrics["primary_outputs"] == 72
        assert metrics["repeated_anchors_exact"] and metrics["trained_anchors_match_FU_exact"]
    for f, h in metrics["bindings"].items():
        assert sha(Path(f)) == h
    extra = read(folder / "observations.json")
    records = read(folder / "manifest.json")["records"]
    assert len(extra) == len(records) == pages
    for r in records:
        assert sha(folder / r["file"]) == r["sha256"] and extra[r["file"]]
scene = read(root / "scene512/metrics.json")
write(
    root / "diagnostics/integrity.json",
    dict(
        parameter_count={m: 21718 for m in METHODS},
        changed_batches=1500,
        unchanged_batches=4500,
        control_original_payloads_exact=True,
        reference_prediction_files_exact=reference_counts,
        last600_update_seconds={
            m: float(np.mean([r["seconds"][m] for r in rows[-600:]])) for m in METHODS
        },
        quantization_output_rms={
            m: dict(
                mean=float(np.mean([r["quantization_output_rms"][m] for r in scene["rows"]])),
                maximum=float(max(r["quantization_output_rms"][m] for r in scene["rows"])),
            )
            for m in REPORT_METHODS
        },
    ),
)
files = [
    "plan.json",
    "source.json",
    "test-results.json",
    "preflight.json",
    "prefix-audit.json",
    "snapshot-audit.json",
    "training/sampling.jsonl",
    "diagnostics/decision.json",
    "diagnostics/review-observations.json",
    "diagnostics/integrity.json",
]
files += [f"driver-{phase}.json" for phase in phases]
files += [
    f"{g}/{n}.json"
    for g in ("evaluation", "stress", "scene512", "width-sweep")
    for n in ("metrics", "visual-review")
]
files += [
    f"diagnostics/{g}/{n}.json"
    for g in ("ambiguity", "geometry")
    for n in ("metrics", "manifest", "observations")
]
write(
    root / "completion.json",
    dict(
        stage="complete_procedural_shape_training",
        steps=STEPS,
        trained_arms=2,
        outputs=9516,
        native_pages_reviewed=34,
        constructed_pair_outputs=12,
        constructed_pair_pages=3,
        geometry_outputs=144,
        geometry_pages=9,
        adoption=False,
        fresh_confirmation=False,
        evidence={f: sha(root / f) for f in files},
    ),
)
print(dict(complete=True, integrity=read(root / "diagnostics/integrity.json")), flush=True)
