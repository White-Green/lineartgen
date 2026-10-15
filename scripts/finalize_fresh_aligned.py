"""Bind complete input-aligned teacher runs and their native visual review."""

import argparse
import json
from pathlib import Path

import numpy as np
from campaign_protocol import read, sha, write
from context_dropout_objective import context_mask
from fresh_aligned_study import CONTROL, METHODS, SAVED, STEPS, check, select

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
root = p.parse_args().root.resolve()
check(root)
rows = [json.loads(x) for x in (root / "training/sampling.jsonl").read_text().splitlines()]
assert len(rows) == STEPS
old_rows = [json.loads(x) for x in (CONTROL / "training/sampling.jsonl").read_text().splitlines()]
eligible = 0
dropped = 0
edges = 0
for step, row in enumerate(rows):
    assert row["step"] == step + 1
    for k, v in select(step).items():
        assert row[k] == v
    previous = old_rows[step]
    assert row["teacher_audit"]["control_inputs_exact"]
    assert row["teacher_audit"]["original_payload_exact"]
    assert row["teacher_audit"]["masks_and_eligibility_exact"]
    assert (
        row["teacher_audit"]["original_payload_sha256"]
        == previous["candidate_payload_sha256"]["fresh"]
    )
    flags = row["merge_eligible"]["aligned"]
    assert flags == previous["merge_eligible"]["fresh"]
    assert row["context_keep"] == previous["context_keep"]
    eligible += sum(flags)
    for m in METHODS:
        assert row["metrics"][m]["merged_fraction"] == sum(flags) / 4
    keep = context_mask(step % 18000, select(step)["depth"], 4, enabled=True)
    assert row["context_keep"] == keep.tolist()
    dropped += int(np.sum(keep[1:] == 0))
    edges += keep[1:].size
assert read(root / "prefix-audit.json")["passed"] and read(root / "preflight.json")["passed"]
audit = read(root / "snapshot-audit.json")
assert audit["raw_ema_exact"] and audit["exports"] == 28
for f, h in audit["bindings"].items():
    assert sha(Path(f)) == h
for step in SAVED[1:]:
    assert read(root / f"training/restore-{step}.json")["full_state_exact"]
phases = [f"train-{s}" for s in SAVED[2:]] + [
    "preflight",
    "evaluation",
    "stress",
    "scene512",
    "width-sweep",
]
for phase in phases:
    assert read(root / f"driver-{phase}.json")["returncode"] == 0
tests = read(root / "test-results.json")
assert tests["passed"]
for f, h in tests["logs"].items():
    assert sha(root / f) == h
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
        assert sha(root / folder / r["file"]) == r["sha256"]
    write(
        root / group / "visual-review.json",
        dict(
            reviewed=True,
            native_scale=True,
            files={r["file"]: r["sha256"] for r in records},
            observations={r["file"]: notes[folder + "/" + r["file"]] for r in records},
        ),
    )
reuse = {}
for group in ("evaluation", "stress"):
    m = read(root / group / "metrics.json")
    assert len(m["rows"]) == 4608 and m["protected_exact"] and m["coverage_exact"]
    bindings = (
        read(root / group / "prediction-binding.json")
        if group == "evaluation"
        else read(root / group / "binding.json")["predictions"]
    )
    for f, h in bindings.items():
        assert sha(root / group / f) == h
    count = 0
    for file in (root / group / "predictions").glob("*.npz"):
        with np.load(file) as a, np.load(CONTROL / group / "predictions" / file.name) as b:
            np.testing.assert_array_equal(a["initial"], b["initial"])
            np.testing.assert_array_equal(a["control"], b["fresh"])
        count += 1
    reuse[group] = count
for group, expected, key in [("width-sweep", 108, "outputs"), ("scene512", 192, "model_outputs")]:
    m = read(root / group / "metrics.json")
    assert m[key] == expected
    for f, h in m["bindings"].items():
        assert sha(Path(f)) == h
assert read(root / "width-sweep/metrics.json")["states_unchanged"]
assert read(root / "scene512/metrics.json")["protected_exact"]
for file in (root / "width-sweep").glob("wide-*.npz"):
    with np.load(file) as a, np.load(CONTROL / "width-sweep" / file.name) as b:
        np.testing.assert_array_equal(a["source"], b["source"])
        np.testing.assert_array_equal(a["initial"], b["initial"])
        np.testing.assert_array_equal(a["control"], b["fresh"])
for file in (root / "scene512").glob("predictions-*.npz"):
    with np.load(file) as a, np.load(CONTROL / "scene512" / file.name) as b:
        for encoding in ("float", "uint8"):
            np.testing.assert_array_equal(a["initial_" + encoding], b["initial_" + encoding])
            np.testing.assert_array_equal(a["control_" + encoding], b["fresh_" + encoding])
scene = read(root / "scene512/metrics.json")
write(
    root / "diagnostics/integrity.json",
    dict(
        all_inputs_masks_eligibility_matched=True,
        teacher_policy="stroke centerlines follow smooth source displacement; widths unchanged",
        warped_components=sum(r["teacher_audit"]["warped_stroke_components"] for r in rows),
        total_components=sum(r["teacher_audit"]["components"] for r in rows),
        eligible_examples=eligible,
        selected_merged_examples={m: eligible for m in METHODS},
        dropped_edges=dropped,
        total_edges=edges,
        reference_prediction_files_exact=reuse,
        last600_update_seconds={
            m: float(np.mean([r["seconds"][m] for r in rows[-600:]])) for m in METHODS
        },
        quantization_output_rms={
            m: dict(
                mean=float(np.mean([r["quantization_output_rms"][m] for r in scene["rows"]])),
                maximum=float(max(r["quantization_output_rms"][m] for r in scene["rows"])),
            )
            for m in METHODS
        },
    ),
)
teacher = read(root / "teacher-preview/visual-review.json")
assert teacher["reviewed"] and len(teacher["files"]) == 2
for f, h in teacher["files"].items():
    assert sha(root / "teacher-preview" / f) == h
files = [
    "teacher-preview/visual-review.json",
    "plan.json",
    "source.json",
    "test-results.json",
    "preflight.json",
    "prefix-audit.json",
    "snapshot-audit.json",
    "training/sampling.jsonl",
    "diagnostics/integrity.json",
    "diagnostics/decision.json",
    "diagnostics/review-observations.json",
]
files += [f"driver-{phase}.json" for phase in phases]
files += [
    f"{g}/{n}.json"
    for g in ("evaluation", "stress", "scene512", "width-sweep")
    for n in ("metrics", "visual-review")
]
write(
    root / "completion.json",
    dict(
        stage="complete_fresh_aligned",
        steps=STEPS,
        trained_arms=1,
        outputs=9516,
        native_pages_reviewed=34,
        teacher_preview_pages_reviewed=2,
        adoption=False,
        fresh_confirmation=False,
        evidence={f: sha(root / f) for f in files},
    ),
)
print(dict(complete=True, integrity=read(root / "diagnostics/integrity.json")), flush=True)
