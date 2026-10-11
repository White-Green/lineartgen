"""Audit the fixed512px geometry experiment after reviewing all five sheets."""

from pathlib import Path

import numpy as np
from campaign_protocol import read, sha, write
from multiscale_velocity_fit import METHODS, PROBE, SAVED, STEPS, TRAIN, check

root = Path("runs/multiscale-velocity-20261011").resolve()
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
assert (
    initial["initial_parameters_exact"]
    and initial["optimizer_zero"]
    and initial["ema_equals_model"]
)
assert read(root / "prefix-audit.json")["all_state_exact"]
assert read(root / "prefix-audit.json")["metrics_exact"]
assert read(root / "prefix-audit.json")["depths"] == list(range(1, 7))
for step in SAVED:
    assert read(root / f"training/restore-audit-{step}.json")["all_state_exact"]
audit = read(root / "snapshot-audit.json")
assert audit["exports"] == 8 and audit["raw_ema_exact"]
for path, digest in audit["bindings"].items():
    assert sha(Path(path)) == digest
metrics = read(root / "evaluation/metrics.json")
assert len(metrics["rows"]) == 200 and metrics["finite"] and metrics["protected_exact"]
expected = {
    (m, i, d, n) for m in METHODS for i in TRAIN + PROBE for d in (1, 3, 6) for n in range(4)
}
expected |= {("initial", i, 3, 0) for i in TRAIN + PROBE}
assert {(r["method"], r["case"], r["depth"], r["draw"]) for r in metrics["rows"]} == expected
assert len(metrics["checks"]) == 176 and len(metrics["retention_checks"]) == 64
for m in METHODS:
    selected = [c for c in metrics["checks"] if c["method"] == m]
    assert len(selected) == 88 and metrics["fit_passed"][m] == all(c["passed"] for c in selected)
assert metrics["retention_passed"] == all(c["passed"] for c in metrics["retention_checks"])
assert len(metrics["bindings"]) == 52
for path, digest in metrics["bindings"].items():
    assert sha(Path(path)) == digest
    with np.load(path) as arrays:
        assert arrays["prediction"].shape == arrays["physical"].shape == (4, 512, 512, 1)
        assert np.isfinite(arrays["physical"]).all()
        np.testing.assert_array_equal(arrays["prediction"], np.clip(arrays["physical"], -1, 1))
review = read(root / "evaluation/visual-review.json")
render = read(root / "evaluation/render-manifest.json")
assert review["reviewed"] and len(review["files"]) == len(render["records"]) == 5
assert render["all_four_fixed_draws"] and render["crops_from_AD"]
for r in render["records"]:
    assert sha(root / "evaluation" / r["file"]) == r["sha256"] == review["files"][r["file"]]
replay = read(root / "replay-audit.json")
assert replay["exact"] and len(replay["files"]) == len(set(replay["files"])) == 25
assert read(root / "control-preflight.json")["eight_final_and_eight_initial_outputs_exact"]
result = dict(
    stage="complete_multiscale_velocity_fitting_diagnostic",
    adoption=False,
    deployable=False,
    numerical_commit=read(root / "source.json")["commit"],
    steps_multiscale_arm=STEPS,
    frozen_control_replay_exact=True,
    parameters=130628,
    trained_depths=list(range(1, 7)),
    fixed_train_pairs=4,
    unfitted_probe_pairs=4,
    fit_failures={
        m: sum(not c["passed"] for c in metrics["checks"] if c["method"] == m) for m in METHODS
    },
    fit_passed=metrics["fit_passed"],
    retention_failures=sum(not c["passed"] for c in metrics["retention_checks"]),
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
            "prefix-audit.json",
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
