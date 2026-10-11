"""Close the projected-endpoint diagnostic after all fixed sheets are reviewed."""

from pathlib import Path

import numpy as np
from campaign_protocol import read, sha, write
from projected_endpoint import METHODS, PROBE, TRAIN, check

root = Path("runs/projected-endpoint-20261011").resolve()
check(root)
tests = read(root / "test-results.json")
assert tests["passed"]
for p, digest in tests["logs"].items():
    assert sha(root / p) == digest
replay = read(root / "replay-audit.json")
assert replay["exact"] and len(replay["files"]) == len(set(replay["files"])) == 52
metrics = read(root / "evaluation/metrics.json")
names = (*METHODS, *[f"{m}_projected" for m in METHODS])
expected = {(m, i, d, n) for m in names for i in TRAIN + PROBE for d in (1, 3, 6) for n in range(4)}
expected |= {("initial", i, 3, 0) for i in TRAIN + PROBE}
assert {(r["method"], r["case"], r["depth"], r["draw"]) for r in metrics["rows"]} == expected
assert len(metrics["rows"]) == 392
assert len(metrics["checks"]) == 352 and len(metrics["retention_checks"]) == 128
assert len(metrics["bindings"]) == 102 and metrics["finite"] and metrics["protected_exact"]
for path, digest in metrics["bindings"].items():
    assert sha(Path(path)) == digest
    with np.load(path) as a:
        assert a["physical"].shape == a["prediction"].shape == (4, 512, 512, 1)
        assert np.isfinite(a["physical"]).all()
        np.testing.assert_array_equal(a["prediction"], np.clip(a["physical"], -1, 1))
trajectory = read(root / "evaluation/trajectory.json")
assert len(trajectory["traces"]) == 8
assert all(len(t["rows"]) == 32 for t in trajectory["traces"])
render = read(root / "evaluation/render-manifest.json")
review = read(root / "evaluation/visual-review.json")
assert review["reviewed"] and len(review["files"]) == len(render["records"]) == 7
for r in render["records"]:
    assert sha(root / "evaluation" / r["file"]) == r["sha256"] == review["files"][r["file"]]
failures = {m: sum(not c["passed"] for c in metrics["checks"] if c["method"] == m) for m in names}
retention = {
    m: sum(not c["passed"] for c in metrics["retention_checks"] if c["method"] == m)
    for m in METHODS
}
result = dict(
    stage="complete_projected_endpoint_diagnostic",
    adoption=False,
    deployable=False,
    updates=0,
    numerical_commit=read(root / "source.json")["commit"],
    fit_failures=failures,
    retention_failures=retention,
    original_replay_exact=True,
    original_blocks=52,
    total_blocks=102,
    final_images_reviewed=7,
    finite=True,
    protected_pixels_exact=True,
    confirmation=False,
    sealed_confirmation_generated=False,
    evidence={
        p: sha(root / p)
        for p in (
            "source.json",
            "plan.json",
            "test-results.json",
            "replay-audit.json",
            "evaluation/metrics.json",
            "evaluation/trajectory.json",
            "evaluation/visual-review.json",
        )
    },
)
write(root / "final-audit.json", result)
write(root / "workflow-status.json", result)
print({k: v for k, v in result.items() if k != "evidence"})
