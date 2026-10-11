"""Finish the teacher-conditioned endpoint posterior diagnostic."""

from collections import Counter
from pathlib import Path

from campaign_protocol import read, sha, write
from endpoint_posterior import METHODS, check

root = Path("runs/endpoint-posterior-20261011").resolve()
check(root)
tests = read(root / "test-results.json")
assert tests["passed"]
for path, digest in tests["logs"].items():
    assert sha(root / path) == digest
for path, digest in read(root / "evaluation/checkpoint-binding.json")["files"].items():
    assert sha(Path(path)) == digest
m = read(root / "evaluation/metrics.json")
assert m["teacher_conditioned"] and m["neural_updates"] == 0
assert len(m["rows"]) == 384 and len(m["bindings"]) == 96
assert Counter(r["method"] for r in m["rows"]) == dict.fromkeys(METHODS, 96)
assert m["finite"] and m["protected_exact"]
assert all(r["finite"] and r["protected_exact"] for r in m["rows"])
assert all(
    r["posterior_mean"] == 0.5 and r["posterior_resolved"] == 0 for r in m["rows"] if r["tau"] == 0
)
for path, digest in m["bindings"].items():
    assert sha(Path(path)) == digest
manifest = read(root / "evaluation/render-manifest.json")
review = read(root / "evaluation/visual-review.json")
assert review["reviewed"] and len(manifest["records"]) == len(review["files"]) == 8
for r in manifest["records"]:
    assert sha(root / "evaluation" / r["file"]) == r["sha256"] == review["files"][r["file"]]
result = dict(
    stage="complete_endpoint_posterior_diagnostic",
    neural_updates=0,
    adoption=False,
    deployable=False,
    teacher_conditioned=True,
    numerical_commit=read(root / "source.json")["commit"],
    rows=384,
    input_groups=96,
    samples_per_group=64,
    final_images_reviewed=8,
    new_teacher_shapes=False,
    confirmation=False,
    sealed_confirmation_generated=False,
    evidence={
        n: sha(root / n)
        for n in (
            "source.json",
            "plan.json",
            "test-results.json",
            "evaluation/checkpoint-binding.json",
            "evaluation/metrics.json",
            "evaluation/render-manifest.json",
            "evaluation/visual-review.json",
        )
    },
)
write(root / "final-audit.json", result)
write(root / "workflow-status.json", result)
print({k: v for k, v in result.items() if k != "evidence"})
