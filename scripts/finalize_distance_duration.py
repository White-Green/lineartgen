"""Audit the fixed512px geometry experiment after reviewing all five sheets."""

from pathlib import Path

import numpy as np
from campaign_protocol import read, sha, write
from distance_duration import METHODS, PROBE, SAVED, START, STEPS, SWITCH, TRAIN, TRAINED, check

root = Path("runs/distance-duration-20261011").resolve()
plan = check(root)
tests = read(root / "test-results.json")
assert tests["passed"]
for name, digest in tests["logs"].items():
    assert sha(root / name) == digest
initial = read(root / "initial-audit.json")
data = read(root / "data-audit.json")
assert data["disjoint"] and data["train"] == list(TRAIN) and data["probe"] == list(PROBE)
assert data["inputs_sha256"] == sha(root / "inputs.npz")
assert len(data["rows"]) == 8 and all(r["antialias_pixels"] > 0 for r in data["rows"])
assert initial["model_optimizer_ema_exact"] and initial["all_depth_outputs_exact"]
assert initial["optimizer_step"] == START
for origin in (START, SWITCH):
    prefix = read(root / f"prefix-audit-{origin}.json")
    assert prefix["all_state_exact"] and prefix["metrics_exact"]
    assert prefix["depths"] == list(range(1, 7))
assert read(root / "phase-switch-audit.json")["all_state_exact"]
for step in SAVED:
    assert read(root / f"training/restore-audit-{step}.json")["all_state_exact"]
audit = read(root / "snapshot-audit.json")
assert audit["exports"] == 20 and audit["raw_ema_exact"]
for path, digest in audit["bindings"].items():
    assert sha(Path(path)) == digest
metrics = read(root / "evaluation/metrics.json")
assert len(metrics["rows"]) == 72 and metrics["finite"] and metrics["protected_exact"]
expected = {(m, i, d) for m in METHODS for i in TRAIN + PROBE for d in (1, 3, 6)}
assert {(r["method"], r["case"], r["depth"]) for r in metrics["rows"]} == expected
assert len(metrics["checks"]) == 66 and len(metrics["retention_checks"]) == 32
for m in METHODS:
    selected = [c for c in metrics["checks"] if c["method"] == m]
    assert len(selected) == 22 and metrics["fit_passed"][m] == all(c["passed"] for c in selected)
for m in TRAINED:
    checks = [c for c in metrics["retention_checks"] if c["method"] == m]
    assert len(checks) == 16
    assert metrics["retention_passed"][m] == all(c["passed"] for c in checks)
assert len(metrics["bindings"]) == 21
for path, digest in metrics["bindings"].items():
    assert sha(Path(path)) == digest
    with np.load(path) as arrays:
        assert arrays["prediction"].shape == (4, 512, 512, 1)
        assert np.isfinite(arrays["prediction"]).all()
        assert np.max(np.abs(arrays["prediction"])) <= 1
review = read(root / "evaluation/visual-review.json")
render = read(root / "evaluation/render-manifest.json")
assert review["reviewed"] and len(review["files"]) == len(render["records"]) == 5
assert render["deterministic_source_model"] and render["crops_from_AD"]
for r in render["records"]:
    assert sha(root / "evaluation" / r["file"]) == r["sha256"] == review["files"][r["file"]]
replay = read(root / "replay-audit.json")
assert replay["exact"] and len(replay["cases"]) == 8 and set(replay["cases"]) == set(TRAIN + PROBE)
assert read(root / "control-preflight.json")["eight_outputs_exact"]
result = dict(
    stage="complete_distance_duration_fitting_diagnostic",
    adoption=False,
    deployable=False,
    numerical_commit=read(root / "source.json")["commit"],
    start_step=START,
    final_step=STEPS,
    additional_updates_per_arm=STEPS - START,
    frozen_control_replay_exact=True,
    parameters=130605,
    trained_depths=list(range(1, 7)),
    fixed_train_pairs=4,
    unfitted_probe_pairs=4,
    fit_failures={
        m: sum(not c["passed"] for c in metrics["checks"] if c["method"] == m) for m in METHODS
    },
    fit_passed=metrics["fit_passed"],
    retention_failures={
        m: sum(not c["passed"] for c in metrics["retention_checks"] if c["method"] == m)
        for m in TRAINED
    },
    methods_advance={
        m: metrics["fit_passed"][m] and metrics["retention_passed"][m] for m in TRAINED
    },
    final_images_reviewed=5,
    full_state_restore_exact=True,
    raw_ema_exports_exact=True,
    finite=True,
    protected_pixels_exact=True,
    confirmation=False,
    sealed_confirmation_generated=False,
    evidence={
        name: sha(root / name)
        for name in (
            "source.json",
            "plan.json",
            "test-results.json",
            "target-oracle.json",
            "data-audit.json",
            "initial-audit.json",
            "control-preflight.json",
            "prefix-audit-1200.json",
            "prefix-audit-6000.json",
            "phase-switch-audit.json",
            "replay-audit.json",
            "snapshot-audit.json",
            "evaluation/metrics.json",
            "evaluation/visual-review.json",
        )
    },
)
write(root / "final-audit.json", result)
write(root / "workflow-status.json", result)
print({k: v for k, v in result.items() if k != "evidence"})
