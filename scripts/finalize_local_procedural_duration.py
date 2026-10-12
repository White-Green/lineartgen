"""Finalize the broad local-CNN diagnostic after numerical and visual review."""

import json
from collections import Counter
from pathlib import Path

from campaign_protocol import read, sha, write
from evaluate_local_procedural_duration import METHODS, folder_name
from local_procedural_duration import AUDIT_STEPS, DATA_START, SAVED, START, STEPS, check

root = Path("runs/local-procedural-duration-20261012").resolve()
plan = check(root)
tests = read(root / "test-results.json")
assert tests["passed"]
for name, digest in tests["logs"].items():
    assert sha(root / name) == digest
initial = read(root / "initial-audit.json")
assert initial["parent_full_state_exact"] and initial["initial_parameters_exact"]
assert initial["optimizer_step"] == START and initial["parameters"] == {"local": 15809}
assert read(root / "backend-audit.json")["flags"] == ""
preflight = read(root / "control-preflight.json")
assert preflight["CL_clipped_blocks_exact"] == 76 and preflight["reference_outputs_exact"] == 16
prefix = read(root / "prefix-audit.json")
assert prefix["all_state_exact"] and prefix["metrics_exact"]
assert prefix["depths"] == [(DATA_START + n) % 6 + 1 for n in range(6)]
for step in SAVED:
    assert read(root / f"training/restore-audit-{step}.json")["all_state_exact"]
audit = read(root / "snapshot-audit.json")
assert audit["exports"] == 8 and audit["raw_ema_exact"]
for path, digest in audit["bindings"].items():
    assert sha(Path(path)) == digest
sampling = [json.loads(s) for s in (root / "training/sampling.jsonl").read_text().splitlines()]
assert [r["step"] for r in sampling] == list(range(START + 1, STEPS + 1))
assert [r["data_step"] for r in sampling] == list(
    range(DATA_START + 1, DATA_START + STEPS - START + 1)
)
assert all(r["depth"] == (r["data_step"] - 1) % 6 + 1 for r in sampling)
assert len({r["inputs_sha256"] for r in sampling}) == STEPS - START
teachers, depths = Counter(), Counter()
assert len(sampling) == 10000
for actual in sampling:
    assert sum(actual["teacher_counts"].values()) == 4
    teachers.update(actual["teacher_counts"])
    depths[actual["depth"]] += 1
assert sum(teachers.values()) == 40000 and "native_corpus" not in teachers
data = read(root / "data-audit.json")
assert len(data["rows"]) == len(AUDIT_STEPS)
for row in data["rows"]:
    actual = sampling[row["step"] - DATA_START - 1]
    assert actual["data_step"] == row["step"]
    assert all(actual[k] == row[k] for k in ("clean_sha256", "teacher_counts"))

evaluations = {}
for steps in (2, 32):
    out = root / folder_name(steps)
    metrics = read(out / "metrics.json")
    assert len(metrics["rows"]) == 6264 and metrics["coverage_exact"] and metrics["protected_exact"]
    assert {(r["case"], r["method"]) for r in metrics["rows"]} == {
        (i, m) for i in range(1566) for m in METHODS
    }
    for path, digest in read(out / "checkpoint-binding.json")["files"].items():
        assert sha(Path(path)) == digest
    replay = read(root / f"replay-audit-{steps}.json")
    assert replay["reference_rows_exact"] == 4698
    summary = read(out / "summary.json")
    assert len(summary["checks"]) == 458
    for m in ("fixed_local", "local"):
        checks = [c for c in summary["checks"] if c["method"] == m]
        assert len(checks) == 229
        assert summary["methods_passed"][m] == all(c["passed"] for c in checks)
    manifest = read(out / "native-review/manifest.json")
    review = read(out / "visual-review.json")
    assert review["reviewed"] and len(review["files"]) == len(manifest["records"]) == 6
    for record in manifest["records"]:
        assert sha(out / "native-review" / record["file"]) == record["sha256"]
        assert record["sha256"] == review["files"][record["file"]]
        for case in record["inputs"]:
            assert (
                sha(out / f"predictions/case-{case['case']:04d}.npz") == case["predictions_sha256"]
            )
    evaluations[steps] = dict(
        failures={
            m: sum(not c["passed"] for c in summary["checks"] if c["method"] == m)
            for m in ("fixed_local", "local")
        },
        methods_passed=summary["methods_passed"],
        evidence={
            name: sha(out / name)
            for name in (
                "metrics.json",
                "summary.json",
                "checkpoint-binding.json",
                "visual-review.json",
                "native-review/manifest.json",
                "black-interiors.json",
            )
        },
    )

result = dict(
    stage="complete_local_procedural_duration_diagnostic",
    adoption=False,
    deployable=False,
    numerical_commit=read(root / "source.json")["commit"],
    start=START,
    final_step=STEPS,
    additional_steps=STEPS - START,
    parameters=15809,
    training_teachers_procedural_only=True,
    teacher_counts=dict(teachers),
    depth_counts=dict(depths),
    independently_audited_teacher_batches=len(AUDIT_STEPS),
    full_state_restore_exact=True,
    raw_ema_exports_exact=True,
    evaluated_rows=12528,
    final_images_reviewed=12,
    evaluations=evaluations,
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
            "replay-audit-2.json",
            "replay-audit-32.json",
            "backend-audit.json",
        )
    },
)
write(root / "final-audit.json", result)
write(root / "workflow-status.json", result)
print({k: v for k, v in result.items() if k != "evidence"})
