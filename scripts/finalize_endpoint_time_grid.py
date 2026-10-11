"""Audit the fixed endpoint sampler comparison after native image review."""

from collections import Counter
from pathlib import Path

from campaign_protocol import read, sha, write
from endpoint_time_grid import ANCHORS, SCHEMES, check

root = Path("runs/endpoint-time-grid-20261011").resolve()
check(root)
assert read(root / "train.json")["neural_updates"] == 0
tests = read(root / "test-results.json")
assert tests["passed"]
for path, digest in tests["logs"].items():
    assert sha(root / path) == digest
replay = read(root / "replay-audit.json")
assert replay["outputs_exact"] == 24 and replay["metrics_exact"]
m = read(root / "evaluation/metrics.json")
assert len(m["rows"]) == len(m["bindings"]) == 72 and len(m["checks"]) == 216
assert m["finite"] and m["protected_exact"] and m["samples_per_group"] == 64
methods = [f"{a}_{s}" for a in ANCHORS for s in SCHEMES]
assert Counter(r["method"] for r in m["rows"]) == dict.fromkeys(methods, 12)
assert Counter(r["method"] for r in m["checks"]) == dict.fromkeys(methods, 36)
for path, digest in m["bindings"].items():
    assert sha(Path(path)) == digest
manifest = read(root / "evaluation/render-manifest.json")
review = read(root / "evaluation/visual-review.json")
assert review["reviewed"] and len(manifest["records"]) == len(review["files"]) == 6
for r in manifest["records"]:
    assert sha(root / "evaluation" / r["file"]) == r["sha256"] == review["files"][r["file"]]
failures = Counter(c["method"] for c in m["checks"] if not c["passed"])
assert failures["noisy_uniform32"] == 22 and failures["source_uniform32"] == 18
result = dict(
    stage="complete_endpoint_time_grid_diagnostic",
    adoption=False,
    deployable=False,
    numerical_commit=read(root / "source.json")["commit"],
    neural_updates=0,
    parameters=130628,
    checkpoint=6000,
    failures={x: failures.get(x, 0) for x in methods},
    checks_per_method=36,
    original_outputs_exact=24,
    evaluated_blocks=72,
    samples_per_group=64,
    final_images_reviewed=6,
    new_teacher_shapes=False,
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
