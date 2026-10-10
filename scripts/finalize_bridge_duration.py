"""Finalize the learned ambiguity toy only after reviewing all three fixed sheets."""

from pathlib import Path

import numpy as np
from bridge_duration import METHODS, SAVED, STEPS, check
from campaign_protocol import read, sha, write

root = Path("runs/bridge-duration-20261010").resolve()
plan = check(root)
tests = read(root / "test-results.json")
assert tests["passed"]
for name, digest in tests["logs"].items():
    assert sha(root / name) == digest
assert read(root / "initial-audit.json")["all_parent_state_exact"]
assert read(root / "initial-audit.json")["schedules_exact"]
assert read(root / "prefix-audit.json")["all_state_exact"]
for step in SAVED:
    assert read(root / f"training/restore-audit-{step}.json")["all_state_exact"]
audit = read(root / "snapshot-audit.json")
assert audit["raw_ema_exact"] and audit["exports"] == 16
for path, digest in audit["bindings"].items():
    assert sha(Path(path)) == digest
metrics = read(root / "evaluation/metrics.json")
assert len(metrics["rows"]) == 252 and metrics["finite"] and metrics["protected_exact"]
expected = {
    (m, d, p, t, s)
    for m in ("direct", "bridge", "velocity", "transport", "boundary", *METHODS)
    for d in (1, 2, 3)
    for p in (False, True)
    for t in (0.1, 0.5)
    for s in (1, 8, 32)
}
assert {
    (r["method"], r["depth"], r["partial"], r["strength"], r["steps"]) for r in metrics["rows"]
} == expected
assert len(metrics["bindings"]) == 252
for path, digest in metrics["bindings"].items():
    path = Path(path)
    assert sha(path) == digest
    with np.load(path) as arrays:
        assert arrays["prediction"].shape == arrays["physical"].shape == (64, 64, 64, 1)
        assert np.isfinite(arrays["physical"]).all()
        np.testing.assert_array_equal(arrays["prediction"], np.clip(arrays["physical"], -1, 1))
parent = Path(plan["control_parent"])
assert metrics["rows"][:180] == read(parent / "evaluation/metrics.json")["rows"]
checks = metrics["checks"]
assert len(checks) == 72
assert all(sum(c["method"] == m for c in checks) == 36 for m in METHODS)
assert metrics["all_passed"] == all(c["passed"] for c in checks)
review = read(root / "evaluation/visual-review.json")
assert review["reviewed"] and len(review["files"]) == 3
render = read(root / "evaluation/render-manifest.json")
assert render["first_four_fixed"] and len(render["records"]) == 3
for record in render["records"]:
    assert (
        sha(root / "evaluation" / record["file"])
        == record["sha256"]
        == review["files"][record["file"]]
    )
out = dict(
    stage="complete_learned_toy_diagnostic",
    adoption=False,
    numerical_commit=read(root / "source.json")["commit"],
    steps_per_arm=STEPS,
    additional_updates_per_arm=24000,
    arms=list(METHODS),
    reused_controls=["direct", "bridge", "velocity", "transport", "boundary"],
    parameters=130628,
    toy_checks=72,
    toy_failures=sum(not c["passed"] for c in checks),
    toy_passed=metrics["all_passed"],
    failures_per_arm={m: sum(not c["passed"] for c in checks if c["method"] == m) for m in METHODS},
    full_state_restore_exact=True,
    raw_ema_exports_exact=True,
    protected_pixels_exact=True,
    final_images_reviewed=3,
    evaluation_groups=12,
    samples_per_group=64,
    saved_output_blocks=252,
    new_teacher_shapes=False,
    confirmation=False,
    sealed_confirmation_generated=False,
    deployable=False,
    evidence={
        name: sha(root / name)
        for name in (
            "source.json",
            "plan.json",
            "test-results.json",
            "initial-audit.json",
            "snapshot-audit.json",
            "prefix-audit.json",
            "evaluation/metrics.json",
            "evaluation/visual-review.json",
        )
    },
)
write(root / "final-audit.json", out)
write(root / "workflow-status.json", out)
print({k: v for k, v in out.items() if k != "evidence"})
