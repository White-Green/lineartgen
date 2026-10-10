"""Bind the final fixed-width comparison after all 14 output sheets are reviewed."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from campaign_protocol import read, sha, write
from train_full_width_paper import check

root = Path("runs/full-width-paper-20261010").resolve()
plan = check(root)
parent = Path(plan["parent"])
tests = read(root / "test-results.json")
assert tests["passed"]
for name, digest in tests["logs"].items():
    assert sha(root / name) == digest
data = read(root / "data-audit.json")
for key in ("crop_resume_exact", "old_clean_hashes_all_exact", "prepared_arrays_all_depths_exact"):
    assert data[key]
assert data["replayed_old_clean_hashes"] == 166
initial = read(root / "initial-six-depth-audit.json")
assert initial["conditions"] == 48 and initial["protected_exact"]
assert initial["all_within_tolerance"] and initial["maximum_error"] <= 1e-5
assert initial["parameter_counts"] == dict(small=130605, wide=517865)
for step in (6, 1000, 4000):
    assert read(root / f"training/restore-audit-{step}.json")["all_state_exact"]
for step in (0, 1000, 4000):
    assert read(root / f"evaluation/step-{step}/control-audit.json")["all_exact"]
for step in (1000, 4000):
    assert read(root / f"broad/step-{step}/control-audit.json")["all_exact"]
assert read(root / "snapshot-audit.json")["raw_ema_exact"]
for path, digest in read(root / "snapshot-audit.json")["bindings"].items():
    assert sha(Path(path)) == digest
for path, digest in read(root / "development/checkpoint-binding.json")["files"].items():
    assert sha(Path(path)) == digest

sampling = list(map(json.loads, (root / "training/sampling.jsonl").read_text().splitlines()))
old_sampling = list(map(json.loads, (parent / "training/sampling.jsonl").read_text().splitlines()))
assert sampling == old_sampling
assert [x["step"] for x in sampling] == list(range(1, 4001))
teachers = Counter()
depths = Counter(str(x["depth"]) for x in sampling)
for x in sampling:
    teachers.update(x["teacher_counts"])
assert dict(teachers) == data["full_run_expected_counts"]
assert dict(depths) == data["depth_counts"]
journal = list(map(json.loads, (root / "training/training.jsonl").read_text().splitlines()))
old_journal = list(map(json.loads, (parent / "training/training.jsonl").read_text().splitlines()))
assert len(journal) == len(old_journal) == 166
for actual, expected in zip(journal, old_journal, strict=True):
    for key in ("step", "depth", "scale_level", "generated", "coverage_generated", "clean_sha256"):
        assert actual[key] == expected[key]
    assert actual["clean_sha256"] == sampling[actual["step"] - 1]["clean_sha256"]

methods = ("initial", "pixel", "band", "wide")
metrics = read(root / "development/metrics.json")
rows = metrics["rows"]
assert len(rows) == 6264 and metrics["coverage_exact"] and metrics["protected_exact"]
assert {(x["case"], x["method"]) for x in rows} == {(i, m) for i in range(1566) for m in methods}
old_rows = read(parent / "development/metrics.json")["rows"]
for name in methods[:3]:
    assert [x for x in rows if x["method"] == name] == [x for x in old_rows if x["method"] == name]
write(
    root / "reference-audit.json",
    dict(
        conditions_per_method=1566,
        all_metrics_exact=True,
        methods=list(methods[:3]),
        parent_sha256=sha(parent / "development/metrics.json"),
    ),
)
with np.load(root / "evaluation/step-0/predictions.npz") as predictions:
    for key in predictions.files:
        if key.startswith("band-"):
            np.testing.assert_allclose(
                predictions[key],
                predictions[key.replace("band-", "wide-", 1)],
                atol=1e-5,
                rtol=0,
            )
with np.load(root / "evaluation/step-4000/predictions.npz") as predictions:
    assert len(predictions.files) == 128
    for method in ("band", "wide"):
        for split in ("train", "probe"):
            for case in ("fit_straight", "fit_thin", "fit_corner", "fit_gap"):
                values = [
                    predictions[k]
                    for k in predictions.files
                    if k.startswith(f"{method}-{split}-{case}-")
                ]
                assert len(values) == 8
                for value in values[1:]:
                    np.testing.assert_array_equal(values[0], value)
assert len(read(root / "broad/step-4000/metrics.json")["rows"]) == 912
for folder, count in (("broad/step-4000", 8), ("development/native-review", 6)):
    review = read(root / folder / "visual-review.json")
    assert review["reviewed"] and len(review["files"]) == count
    for name, digest in review["files"].items():
        assert sha(root / folder / name) == digest

summary = read(root / "development/summary.json")
assert len(summary["checks"]) == 229
assert summary["failures"] == [x for x in summary["checks"] if not x["passed"]]
passed = summary["methods_passed"]
assert passed == {"wide": all(x["passed"] for x in summary["checks"])}
assert all(x.get("strength", 0.1) in (0.1, 0.5) for x in summary["checks"])
out = dict(
    stage=(
        "complete_development_candidate_pending_confirmation"
        if passed["wide"]
        else "complete_diagnostic_insufficient"
    ),
    adoption=False,
    numerical_commit=read(root / "source.json")["commit"],
    steps_per_arm=4000,
    trained_methods=["wide"],
    reused_controls=["band"],
    methods_passed=passed,
    local_primary_conditions=128,
    local_broad_conditions=912,
    development_conditions=6264,
    development_checks=229,
    development_failures=len(summary["failures"]),
    local_diagnostic_failures=len(read(root / "broad/step-4000/summary.json")["failures"]),
    parameter_counts=initial["parameter_counts"],
    initial_max_error=initial["maximum_error"],
    full_sampling_exact=True,
    depth_counts=dict(depths),
    teacher_counts=dict(teachers),
    full_state_restore_exact=True,
    raw_ema_exports_exact=True,
    protected_pixels_exact=True,
    final_images_reviewed=14,
    gaussian_draws_identical=True,
    evaluation_source_counts=plan["evaluation_source_counts"],
    sealed_confirmation_generated=False,
    initial_and_control_reference_all_metrics_exact=True,
    evidence={
        name: sha(root / name)
        for name in (
            "source.json",
            "plan.json",
            "snapshot-audit.json",
            "data-audit.json",
            "test-results.json",
            "initial-six-depth-audit.json",
            "development/metrics.json",
            "development/summary.json",
            "development/black-interiors.json",
            "development/native-review/visual-review.json",
            "broad/step-4000/visual-review.json",
            "reference-audit.json",
        )
    },
)
write(root / "final-audit.json", out)
write(root / "workflow-status.json", out)
print({k: v for k, v in out.items() if k != "evidence"})
