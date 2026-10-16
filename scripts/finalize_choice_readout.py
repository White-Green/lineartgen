"""Bind complete input-aligned teacher runs and their native visual review."""

import argparse
import json
from pathlib import Path

import numpy as np
from campaign_protocol import read, sha, write
from choice_readout_study import (
    CONTROL,
    LOCALCONTROL,
    METHODS,
    MODELREF,
    OFFSET,
    REPORT_METHODS,
    REUSED,
    SAVED,
    SOURCE,
    STEPS,
    check,
    frequency_plan,
    select,
)
from context_dropout_objective import context_mask

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
root = p.parse_args().root.resolve()
check(root)
rows = [json.loads(x) for x in (root / "training/sampling.jsonl").read_text().splitlines()]
assert len(rows) == STEPS
old_rows = [json.loads(x) for x in (CONTROL / "training/sampling.jsonl").read_text().splitlines()]
source_rows = [json.loads(x) for x in (SOURCE / "training/sampling.jsonl").read_text().splitlines()]
eligible = 0
dropped = 0
edges = 0
for step, row in enumerate(rows):
    assert row["step"] == step + 1
    for k, v in select(step).items():
        assert row[k] == v
    source_step = frequency_plan()["indices"][step]
    previous = old_rows[source_step + OFFSET]
    assert row["teacher_audit"]["control_inputs_exact"]
    assert row["teacher_audit"]["original_payload_exact"]
    assert row["teacher_audit"]["masks_and_eligibility_exact"]
    assert row["teacher_audit"]["aligned_payload_exact"]
    flags = row["merge_eligible"]["choice"]
    assert flags == previous["merge_eligible"]["aligned"]
    assert row["context_keep"] == previous["context_keep"]
    eligible += sum(flags)
    for m in METHODS:
        assert row["metrics"][m]["merged_fraction"] == sum(flags) / 4
        assert row["merge_eligible"][m] == flags
        assert row["candidate_payload_sha256"][m] == previous["candidate_payload_sha256"]["aligned"]
        assert row["input_sha256"][m] == source_rows[source_step]["input_sha256"]["expanded"]
    keep = context_mask((source_step + OFFSET) % 18000, select(step)["depth"], 4, enabled=True)
    assert row["context_keep"] == keep.tolist()
    dropped += int(np.sum(keep[1:] == 0))
    edges += keep[1:].size
assert read(root / "prefix-audit.json")["passed"] and read(root / "preflight.json")["passed"]
audit = read(root / "snapshot-audit.json")
assert audit["raw_ema_exact"] and audit["exports"] == 16
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
    "ambiguity",
    "orientation",
]
for phase in phases:
    assert read(root / f"driver-{phase}.json")["returncode"] == 0
tests = read(root / "test-results.json")
assert tests["passed"]
for f, h in tests["logs"].items():
    assert sha(root / f) == h
widths = read(root / "width-plan.json")
for row, expected in zip(rows, widths, strict=True):
    assert row["teacher_audit"]["source_widths"] == expected["records"]
    assert row["teacher_audit"]["expanded_source_exact"]
    assert any(row["teacher_audit"]["changed_examples"])
    assert row["teacher_audit"]["source_sampling_step"] == row["source_sampling_step"]
    assert row["teacher_audit"]["candidate_teachers_exact"]
assert all(r["metrics"][m]["opening_consistency"] >= 0 for r in rows for m in METHODS)
assert read(root / "frequency-plan.json") == frequency_plan()
ft_rows = [json.loads(x) for x in (REUSED / "training/sampling.jsonl").read_text().splitlines()]
for row, ft in zip(rows, ft_rows, strict=True):
    assert row["context_keep"] == ft["context_keep"]
    for method in METHODS:
        assert row["input_sha256"][method] == ft["input_sha256"]["frequent"]
        assert row["candidate_payload_sha256"][method] == ft["candidate_payload_sha256"]["frequent"]

fv_rows = [
    json.loads(x) for x in (LOCALCONTROL / "training/sampling.jsonl").read_text().splitlines()
]
for row, old in zip(rows, fv_rows, strict=True):
    assert row["context_keep"] == old["context_keep"]
    assert row["input_sha256"]["choice"] == old["input_sha256"]["mixed"]
    assert row["candidate_payload_sha256"]["choice"] == old["candidate_payload_sha256"]["mixed"]
reuse_control = read(root / "control-reuse.json")
for f, h in reuse_control["files"].items():
    assert sha(root / "snapshots/step-6000/mixed/raw" / f) == h
    assert sha(Path(reuse_control["path"]) / f) == h
notes = read(root / "diagnostics/review-observations.json")
source_review = read(root / "source-preview/visual-review.json")
assert source_review["reviewed"]
for f, h in source_review["files"].items():
    assert sha(root / "source-preview" / f) == h
for f, h in read(root / "source-preview/manifest.json")["arrays"].items():
    assert sha(root / "source-preview" / f) == h
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
        with np.load(file) as a, np.load(MODELREF / group / "predictions" / file.name) as b:
            np.testing.assert_array_equal(a["initial"], b["opening"])
            with np.load(LOCALCONTROL / group / "predictions" / file.name) as old:
                np.testing.assert_array_equal(a["mixed"], old["mixed"])
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
    with np.load(file) as a, np.load(MODELREF / "width-sweep" / file.name) as b:
        np.testing.assert_array_equal(a["source"], b["source"])
        np.testing.assert_array_equal(a["initial"], b["opening"])
        with np.load(LOCALCONTROL / "width-sweep" / file.name) as old:
            np.testing.assert_array_equal(a["mixed"], old["mixed"])
for file in (root / "scene512").glob("predictions-*.npz"):
    with np.load(file) as a, np.load(MODELREF / "scene512" / file.name) as b:
        for encoding in ("float", "uint8"):
            np.testing.assert_array_equal(a["initial_" + encoding], b["opening_" + encoding])
            with np.load(LOCALCONTROL / "scene512" / file.name) as old:
                np.testing.assert_array_equal(a["mixed_" + encoding], old["mixed_" + encoding])
scene = read(root / "scene512/metrics.json")
write(
    root / "diagnostics/integrity.json",
    dict(
        resampled_batches_exact_to_source=True,
        recursive_depths_matched=True,
        no_example_reweighting=True,
        pools=frequency_plan()["pool_sizes"],
        both_arms_inputs_exact_to_FT=True,
        parameter_count={"mixed": 21718, "choice": 21718},
        added_parameters=0,
        head_readout="single contribution0 at every level",
        branch_dilations={"mixed": [1, 32], "choice": [1, 32]},
        changed_examples=sum(sum(r["teacher_audit"]["changed_examples"]) for r in rows),
        opening_prior_active=True,
        opening_weight=0.1,
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
            for m in REPORT_METHODS
        },
    ),
)
teacher = read(root / "teacher-preview/visual-review.json")
assert teacher["reviewed"] and len(teacher["files"]) == 2
for f, h in teacher["files"].items():
    assert sha(root / "teacher-preview" / f) == h
ambiguity = read(root / "diagnostics/ambiguity/metrics.json")
assert ambiguity["outputs"] == 12 and ambiguity["states_unchanged"]
for f, h in ambiguity["bindings"].items():
    assert sha(Path(f)) == h
for depth in (1, 3, 6):
    with (
        np.load(root / f"diagnostics/ambiguity/depth-{depth}.npz") as a,
        np.load(LOCALCONTROL / f"diagnostics/ambiguity/depth-{depth}.npz") as b,
    ):
        np.testing.assert_array_equal(a["mixed"], b["mixed"])
ambiguity_notes = read(root / "diagnostics/ambiguity/observations.json")
assert len(ambiguity_notes) == 3
for record in read(root / "diagnostics/ambiguity/manifest.json")["records"]:
    assert sha(root / "diagnostics/ambiguity" / record["file"]) == record["sha256"]
    assert ambiguity_notes[record["file"]]
orientation = read(root / "diagnostics/orientation/metrics.json")
assert orientation["outputs"] == 32 and orientation["states_unchanged"]
for f, h in orientation["bindings"].items():
    assert sha(Path(f)) == h
orientation_notes = read(root / "diagnostics/orientation/observations.json")
orientation_pages = read(root / "diagnostics/orientation/manifest.json")["records"]
assert len(orientation_notes) == len(orientation_pages) == 4
for record in orientation_pages:
    assert sha(root / "diagnostics/orientation" / record["file"]) == record["sha256"]
    assert orientation_notes[record["file"]]
files = [
    "diagnostics/orientation/metrics.json",
    "diagnostics/orientation/manifest.json",
    "diagnostics/orientation/observations.json",
    "diagnostics/ambiguity/metrics.json",
    "diagnostics/ambiguity/manifest.json",
    "diagnostics/ambiguity/observations.json",
    "control-reuse.json",
    "source-preview/visual-review.json",
    "source-preview/manifest.json",
    "teacher-preview/visual-review.json",
    "plan.json",
    "width-plan.json",
    "width-range-audit.json",
    "frequency-plan.json",
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
        stage="complete_choice_readout",
        steps=STEPS,
        trained_arms=1,
        reused_control_arms=1,
        outputs=9516,
        native_pages_reviewed=34,
        orientation_outputs=32,
        orientation_native_pages_reviewed=4,
        constructed_pair_outputs=12,
        constructed_pair_native_pages_reviewed=3,
        teacher_preview_pages_inherited=2,
        adoption=False,
        fresh_confirmation=False,
        evidence={f: sha(root / f) for f in files},
    ),
)
print(dict(complete=True, integrity=read(root / "diagnostics/integrity.json")), flush=True)
