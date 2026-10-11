"""Finish the resolution-order diagnostic after all ten sheet reviews."""

from collections import Counter
from pathlib import Path

import numpy as np
from campaign_protocol import read, sha, write
from cascade_endpoint_eval import ANCHORS, METHODS, PROBE, SCHEMES, TRAIN, check

root = Path("runs/cascade-endpoint-20261011").resolve()
check(root)
assert read(root / "train.json")["neural_updates"] == 0
tests = read(root / "test-results.json")
assert tests["passed"]
for p, h in tests["logs"].items():
    assert sha(root / p) == h
m = read(root / "evaluation/metrics.json")
assert len(m["rows"]) == 576 and len(m["checks"]) == 528 and len(m["retention_checks"]) == 256
assert len(m["bindings"]) == 150 and m["finite"] and m["protected_exact"]
assert {(r["method"], r["case"], r["depth"], r["draw"]) for r in m["rows"]} == {
    (a, i, d, n) for a in METHODS for i in TRAIN + PROBE for d in (1, 3, 6) for n in range(4)
}
assert Counter(c["method"] for c in m["checks"]) == dict.fromkeys(METHODS, 88)
for p, h in m["bindings"].items():
    assert sha(Path(p)) == h
    with np.load(p) as z:
        assert z["physical"].shape == z["prediction"].shape == (4, 512, 512, 1)
        assert np.isfinite(z["physical"]).all()
        np.testing.assert_array_equal(z["prediction"], np.clip(z["physical"], -1, 1))
replay = read(root / "replay-audit.json")
assert (
    replay["exact"]
    and len(replay["files"]) == len(set(replay["files"])) == 50
    and replay["metric_rows_exact"] == 192
)
manifest = read(root / "evaluation/render-manifest.json")
review = read(root / "evaluation/visual-review.json")
assert review["reviewed"] and len(review["files"]) == len(manifest["records"]) == 10
for r in manifest["records"]:
    assert sha(root / "evaluation" / r["file"]) == r["sha256"] == review["files"][r["file"]]
failures = Counter(c["method"] for c in m["checks"] if not c["passed"])
assert failures["control_recursive"] == 58 and failures["band_recursive"] == 53
retention = Counter(c["method"] for c in m["retention_checks"] if not c["passed"])
result = dict(
    stage="complete_cascade_endpoint_diagnostic",
    adoption=False,
    deployable=False,
    numerical_commit=read(root / "source.json")["commit"],
    neural_updates=0,
    parameters=134688,
    checkpoint=7200,
    schemes=list(SCHEMES),
    fit_failures={m: failures.get(m, 0) for m in METHODS},
    retention_failures={
        f"{a}_{n}_vs_recursive": retention.get(f"{a}_{n}", 0)
        for a in ANCHORS
        for n in ("cascade", "oracle")
    },
    checks_per_method=88,
    original_outputs_exact=50,
    original_metric_rows_exact=192,
    evaluated_blocks=150,
    final_images_reviewed=10,
    oracle_teacher_assisted=True,
    confirmation=False,
    sealed_confirmation_generated=False,
    evidence={
        n: sha(root / n)
        for n in (
            "source.json",
            "plan.json",
            "test-results.json",
            "replay-audit.json",
            "evaluation/metrics.json",
            "evaluation/render-manifest.json",
            "evaluation/visual-review.json",
        )
    },
)
write(root / "final-audit.json", result)
write(root / "workflow-status.json", result)
print({k: v for k, v in result.items() if k != "evidence"})
