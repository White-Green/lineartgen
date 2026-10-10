"""Finalize the learned ambiguity toy only after reviewing all three fixed sheets."""

from pathlib import Path

import numpy as np
from bridge_new_noise import METHODS, SAMPLES, check
from campaign_protocol import read, sha, write

root = Path("runs/bridge-new-noise-20261010").resolve()
plan = check(root)
tests = read(root / "test-results.json")
assert tests["passed"]
for name, digest in tests["logs"].items():
    assert sha(root / name) == digest
metrics = read(root / "evaluation/metrics.json")
assert len(metrics["rows"]) == 24 and metrics["finite"] and metrics["protected_exact"]
expected = {
    (m, d, p, t, s)
    for m in METHODS
    for d in (1, 2, 3)
    for p in (False, True)
    for t in (0.1, 0.5)
    for s in (32,)
}
assert {
    (r["method"], r["depth"], r["partial"], r["strength"], r["steps"]) for r in metrics["rows"]
} == expected
assert len(metrics["bindings"]) == 24
for path, digest in metrics["bindings"].items():
    path = Path(path)
    assert sha(path) == digest
    with np.load(path) as arrays:
        assert arrays["prediction"].shape == arrays["physical"].shape == (SAMPLES, 64, 64, 1)
        assert np.isfinite(arrays["physical"]).all()
        np.testing.assert_array_equal(arrays["prediction"], np.clip(arrays["physical"], -1, 1))
keys = read(root / "keys.json")
assert len(keys["keys"]) == 192 and len({tuple(k["key"]) for k in keys["keys"]}) == 192
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
    stage="complete_new_noise_diagnostic",
    adoption=False,
    numerical_commit=read(root / "source.json")["commit"],
    neural_updates=0,
    fixed_weight_step=36000,
    arms=list(METHODS),
    parameters=130628,
    toy_checks=72,
    toy_failures=sum(not c["passed"] for c in checks),
    toy_passed=metrics["all_passed"],
    failures_per_arm={m: sum(not c["passed"] for c in checks if c["method"] == m) for m in METHODS},
    new_noise_seed=keys["seed"],
    protected_pixels_exact=True,
    final_images_reviewed=3,
    evaluation_groups=12,
    samples_per_group=SAMPLES,
    saved_output_blocks=24,
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
            "keys.json",
            "evaluation/metrics.json",
            "evaluation/visual-review.json",
        )
    },
)
write(root / "final-audit.json", out)
write(root / "workflow-status.json", out)
print({k: v for k, v in out.items() if k != "evidence"})
