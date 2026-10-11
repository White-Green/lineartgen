"""Verify broad synthetic training and all reviewed development evidence."""

import json
from collections import Counter
from pathlib import Path

from aux_distance_generalization import AUDIT_STEPS, METHODS, SAVED, STEPS, check
from campaign_protocol import read, sha, write

root = Path("runs/aux-distance-generalization-20261011").resolve()
plan = check(root)
tests = read(root / "test-results.json")
assert tests["passed"]
for name, digest in tests["logs"].items():
    assert sha(root / name) == digest
initial = read(root / "initial-audit.json")
assert (
    initial["full_parameters_identical"]
    and initial["optimizer_zero"]
    and initial["ema_equals_model"]
)
assert read(root / "control-preflight.json")["reference_outputs_exact"] == 16
prefix = read(root / "prefix-audit.json")
assert prefix["all_state_exact"] and prefix["metrics_exact"] and prefix["inputs_regenerated"]
for step in SAVED:
    assert read(root / f"training/restore-audit-{step}.json")["all_state_exact"]
audit = read(root / "snapshot-audit.json")
assert audit["exports"] == 16 and audit["raw_ema_exact"]
for path, digest in audit["bindings"].items():
    assert sha(Path(path)) == digest
sampling = [json.loads(s) for s in (root / "training/sampling.jsonl").read_text().splitlines()]
assert [r["step"] for r in sampling] == list(range(1, STEPS + 1))
assert all(r["depth"] == (r["step"] - 1) % 6 + 1 for r in sampling)
assert all(sum(r["teacher_counts"].values()) == 4 for r in sampling)
depths, teachers = Counter(), Counter()
for row in sampling:
    depths[row["depth"]] += 1
    teachers.update(row["teacher_counts"])
assert sum(teachers.values()) == 16000 and "native_corpus" not in teachers
data = read(root / "data-audit.json")
assert len(data["rows"]) == len(AUDIT_STEPS)
for r in data["rows"]:
    actual = sampling[r["step"] - 1]
    assert all(actual[k] == r[k] for k in ("step", "clean_sha256", "teacher_counts"))
metrics = read(root / "development/metrics.json")
assert len(metrics["rows"]) == 6264 and metrics["coverage_exact"] and metrics["protected_exact"]
for path, digest in read(root / "development/checkpoint-binding.json")["files"].items():
    assert sha(Path(path)) == digest
replay = read(root / "replay-audit.json")
assert replay["reference_rows_exact"] == 3132
summary = read(root / "development/summary.json")
assert len(summary["checks"]) == 458
assert all(len([c for c in summary["checks"] if c["method"] == m]) == 229 for m in METHODS)
manifest = read(root / "development/native-review/manifest.json")
review = read(root / "development/visual-review.json")
assert review["reviewed"] and len(review["files"]) == len(manifest["records"]) == 6
for row in manifest["records"]:
    assert (
        sha(root / "development/native-review" / row["file"])
        == row["sha256"]
        == review["files"][row["file"]]
    )
    for case in row["inputs"]:
        assert (
            sha(root / f"development/predictions/case-{case['case']:04d}.npz")
            == case["predictions_sha256"]
        )
result = dict(
    stage="complete_aux_distance_generalization_diagnostic",
    adoption=False,
    deployable=False,
    numerical_commit=read(root / "source.json")["commit"],
    steps_per_arm=STEPS,
    inference_parameters=130605,
    training_parameters=134382,
    training_teachers_procedural_only=True,
    teacher_counts=dict(teachers),
    depth_counts=dict(depths),
    audited_target_hashes=len(AUDIT_STEPS),
    full_state_restore_exact=True,
    raw_ema_exports_exact=True,
    frozen_reference_rows_exact=3132,
    evaluated_rows=6264,
    final_images_reviewed=6,
    failures={
        m: sum(not c["passed"] for c in summary["checks"] if c["method"] == m) for m in METHODS
    },
    methods_passed=summary["methods_passed"],
    confirmation=False,
    sealed_confirmation_generated=False,
    evidence={
        name: sha(root / name)
        for name in (
            "source.json",
            "plan.json",
            "test-results.json",
            "data-audit.json",
            "initial-audit.json",
            "control-preflight.json",
            "prefix-audit.json",
            "snapshot-audit.json",
            "training/sampling.jsonl",
            "replay-audit.json",
            "development/metrics.json",
            "development/summary.json",
            "development/visual-review.json",
        )
    },
)
write(root / "final-audit.json", result)
write(root / "workflow-status.json", result)
print({k: v for k, v in result.items() if k != "evidence"})
