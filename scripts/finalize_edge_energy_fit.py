from collections import Counter
from pathlib import Path

import numpy as np
from campaign_protocol import read, sha, write
from edge_energy_fit import METHODS, PROBE, TRAIN, check, fit_checks

r = Path("runs/edge-energy-fit-20261010").resolve()
plan = check(r)
assert read(r / "gate-audit.json")["allowed"]
assert read(r / "test-results.json")["passed"]
for name, digest in read(r / "test-results.json")["logs"].items():
    assert sha(r / name) == digest
initial = read(r / "initial-audit.json")
assert initial["all_depths_within_1e_minus_5"]
assert len(initial["errors"]) == 6 and all(x["maximum_error"] <= 1e-5 for x in initial["errors"])
assert initial["parameters"]["energy"] == 130605
assert all(x["passed"] for x in read(r / "target-oracle.json")["checks"])
for step in (6, 100, 400, 1200):
    assert read(r / f"training/restore-audit-{step}.json")["all_state_exact"]
exports = read(r / "snapshot-audit.json")
assert exports["raw_ema_exact"] and exports["exports"] == 6
for path, digest in exports["bindings"].items():
    assert sha(Path(path)) == digest
trajectory = {}
for step in (0, 100, 400, 1200):
    p = r / f"evaluation/step-{step}"
    j = read(p / "metrics.json")
    rows = j["rows"]
    assert read(p / "control-audit.json")["all_exact"]
    parent = Path(plan["control_parent"])
    old = [
        x
        for x in read(parent / f"evaluation/step-{step}/metrics.json")["rows"]
        if x["method"] == "band"
    ]
    assert [x for x in rows if x["method"] == "band"] == old
    assert len(rows) == 16
    assert {(x["method"], x["case"]) for x in rows} == {
        (m, i) for m in METHODS for i in TRAIN + PROBE
    }
    assert j["checks"] == fit_checks(rows)
    with np.load(p / "predictions.npz") as pred:
        assert len(pred.files) == 16 and all(np.isfinite(pred[k]).all() for k in pred.files)
    coverage = read(p / "coverage-metrics.json")["rows"]
    assert len(coverage) == 16
    assert {(x["method"], x["case"]) for x in coverage} == {
        (m, i) for m in METHODS for i in TRAIN + PROBE
    }
    assert all(0 <= x["coverage_overlap_f1"] <= 1 for x in coverage)
    trajectory[str(step)] = dict(
        fit_passed=j["fit_passed"],
        failures=dict(Counter(x["method"] for x in j["checks"] if not x["passed"])),
        metrics_sha256=sha(p / "metrics.json"),
        predictions_sha256=sha(p / "predictions.npz"),
    )
review = read(r / "evaluation/step-1200/visual-review.json")
assert review["reviewed"] and len(review["files"]) == 2
for name, digest in review["files"].items():
    assert sha(r / "evaluation/step-1200" / name) == digest
gate = read(r / "gate-audit.json")
parent = Path(plan["control_parent"])
assert gate["parent_final_audit_sha256"] == sha(parent / "final-audit.json")
m = r / "energy-mechanism"
assert gate["mechanism_audit_sha256"] == sha(m / "audit.json")
mechanism = read(m / "audit.json")
for key in [
    "finite_losses",
    "direct_descent_correct",
    "protected_gradients_zero",
    "target_gradient_zero",
]:
    assert mechanism[key]
assert mechanism["config_only_differences"] == ["loss.edge_energy_weight"]
for name, digest in mechanism["files"].items():
    assert sha(m / name) == digest
for path, digest in mechanism["source_bindings"].items():
    assert sha(Path(path)) == digest
final_rows = read(r / "evaluation/step-1200/metrics.json")["rows"]
lookup = {(x["method"], x["case"]): x for x in final_rows}
regression_checks = []
for i in TRAIN:
    for metric, threshold, lower in [
        ("ink_f1_alpha50", -0.005, False),
        ("ink_deficit_fraction", 0.01, True),
    ]:
        difference = lookup["energy", i][metric] - lookup["band", i][metric]
        regression_checks.append(
            dict(
                case=i,
                metric=metric,
                difference=difference,
                threshold=threshold,
                passed=difference <= threshold if lower else difference >= threshold,
            )
        )
write(
    r / "regression-checks.json",
    dict(checks=regression_checks, passed=all(x["passed"] for x in regression_checks)),
)
out = dict(
    stage="complete_fitting_diagnostic",
    adoption=False,
    numerical_commit=read(r / "source.json")["commit"],
    steps_per_arm=1200,
    arms=["energy"],
    reused_controls=["band"],
    parameters=initial["parameters"],
    fit_passed=trajectory["1200"]["fit_passed"],
    trajectory=trajectory,
    teacher_identity_checks_passed=True,
    train_regression_checks_passed=all(x["passed"] for x in regression_checks),
    initial_max_error=max(x["maximum_error"] for x in initial["errors"]),
    full_state_restore_exact=True,
    raw_ema_exports_exact=True,
    final_images_reviewed=2,
    train_cases=list(TRAIN),
    probe_cases=list(PROBE),
    trained_on_development_geometry=True,
    confirmation=False,
    sealed_confirmation_generated=False,
    evidence={
        n: sha(r / n)
        for n in [
            "source.json",
            "plan.json",
            "test-results.json",
            "target-oracle.json",
            "initial-audit.json",
            "snapshot-audit.json",
            "gate-audit.json",
            "regression-checks.json",
            "energy-mechanism/audit.json",
            "evaluation/step-1200/visual-review.json",
            "evaluation/step-1200/coverage-metrics.json",
        ]
    },
)
out["advance_to_broad_training"] = bool(
    out["fit_passed"]["energy"] and out["train_regression_checks_passed"]
)
write(r / "final-audit.json", out)
write(r / "workflow-status.json", out)
print(out)
