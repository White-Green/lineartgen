"""Bind the matched full-data residual comparison and its actual visual review."""

import argparse
import json
from pathlib import Path

import numpy as np
from campaign_protocol import read, sha, write
from residual_axis_study import (
    GF,
    GG,
    LOCALCONTROL,
    METHODS,
    REPORT_METHODS,
    SAVED,
    STEPS,
    check,
    select,
)

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
root = p.parse_args().root.resolve()
check(root)
rows = [json.loads(x) for x in (root / "training/sampling.jsonl").read_text().splitlines()]
old = [json.loads(x) for x in (LOCALCONTROL / "training/sampling.jsonl").read_text().splitlines()]
assert len(rows) == STEPS
eligible = []
for step, (a, b) in enumerate(zip(rows, old, strict=True)):
    assert a["step"] == step + 1
    for k, v in select(step).items():
        assert a[k] == v
    assert a["context_keep"] == b["context_keep"]
    for m in METHODS:
        assert a["input_sha256"][m] == b["input_sha256"]["mixed"]
        assert a["candidate_payload_sha256"][m] == b["candidate_payload_sha256"]["mixed"]
        assert a["merge_eligible"][m] == b["merge_eligible"]["mixed"]
        assert a["metrics"][m]["axis_eligible_fraction"] == sum(a["axis_allowed"]) / 4
        assert a["metrics"][m]["merged_fraction"] == sum(a["merge_eligible"][m]) / 4
    for item, allowed in enumerate(a["axis_allowed"]):
        record = a["teacher_audit"]["axis_records"][item]
        assert record["allowed"] == allowed
        if allowed:
            assert a["modes"][item] != 2 and not a["overlays"][item]["enabled"]
            assert record["ratio"] >= 0.75
            eligible.append((step, item, a["indices"][item], a["depth"]))
expected = [
    (r["step"], r["item"], r["index"], r["depth"])
    for r in read(GG / "eligibility.json")["examples"]
]
assert eligible == expected and len(eligible) == 120
assert read(root / "preflight.json")["passed"] and read(root / "prefix-audit.json")["passed"]
audit = read(root / "snapshot-audit.json")
assert audit["raw_ema_exact"] and audit["exports"] == 32
for f, h in audit["bindings"].items():
    assert sha(Path(f)) == h
for step in SAVED[1:]:
    assert read(root / f"training/restore-{step}.json")["full_state_exact"]
phases = [
    "preflight",
    *[f"train-{s}" for s in SAVED[2:]],
    "evaluation",
    "stress",
    "scene512",
    "width-sweep",
    "ambiguity",
    "orientation",
]
for phase in phases:
    assert read(root / f"driver-{phase}.json")["returncode"] == 0
assert read(root / "test-results.json")["passed"]
for folder in ("source-preview", "teacher-preview"):
    review = read(root / folder / "visual-review.json")
    assert review["reviewed"]
    for f, h in review["files"].items():
        assert sha(root / folder / f) == h
assert read(GF / "decision.json")["mechanism_passed"]
assert read(root / "axis-preview/decision.json")["teacher_preview_passed"]
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
for group, outputs, pages in [("ambiguity", 12, 3), ("orientation", 32, 4)]:
    folder = root / "diagnostics" / group
    metrics = read(folder / "metrics.json")
    assert metrics["outputs"] == outputs and metrics["states_unchanged"]
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
        parameter_count={m: 21830 for m in METHODS},
        eligible=120,
        matched_original_payloads=True,
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
    for g in ("ambiguity", "orientation")
    for n in ("metrics", "manifest", "observations")
]
write(
    root / "completion.json",
    dict(
        stage="complete_residual_axis",
        steps=STEPS,
        trained_arms=2,
        outputs=9516,
        native_pages_reviewed=34,
        constructed_pair_outputs=12,
        constructed_pair_pages=3,
        orientation_outputs=32,
        orientation_pages=4,
        adoption=False,
        fresh_confirmation=False,
        evidence={f: sha(root / f) for f in files},
    ),
)
print(dict(complete=True, integrity=read(root / "diagnostics/integrity.json")), flush=True)
