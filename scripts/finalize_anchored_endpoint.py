"""Finish the endpoint-anchor toy after all numerical and visual audits."""

import json
from collections import Counter
from pathlib import Path

from anchored_endpoint_fit import METHODS, SAVED, STEPS, check
from campaign_protocol import read, sha, write

root = Path("runs/anchored-endpoint-20261011").resolve()
plan = check(root)
tests = read(root / "test-results.json")
assert tests["passed"]
for path, digest in tests["logs"].items():
    assert sha(root / path) == digest
a = read(root / "initial-audit.json")
assert a["full_parameters_identical"] and a["optimizer_zero"] and a["ema_equals_model"]
a = read(root / "control-preflight.json")
assert a["parent_parameters_exact"] and a["initial_noisy_endpoint_exact"]
a = read(root / "prefix-audit.json")
assert a["all_state_exact"] and a["metrics_exact"]
for step in SAVED:
    assert read(root / f"training/restore-audit-{step}.json")["all_state_exact"]
a = read(root / "snapshot-audit.json")
assert a["raw_ema_exact"] and a["exports"] == 16
for path, digest in a["bindings"].items():
    assert sha(Path(path)) == digest
sampling = [json.loads(s) for s in (root / "training/sampling.jsonl").read_text().splitlines()]
assert [r["step"] for r in sampling] == list(range(1, STEPS + 1))
assert all(r["depth"] == (r["step"] - 1) % 3 + 1 and r["label_A"] == 8 for r in sampling)
assert len({r["inputs_sha256"] for r in sampling}) == STEPS
parent_sampling = [
    json.loads(s)
    for s in (Path(plan["data_parent"]) / "training/sampling.jsonl").read_text().splitlines()
]
assert sampling == parent_sampling
for path, digest in read(root / "evaluation/checkpoint-binding.json")["files"].items():
    assert sha(Path(path)) == digest
m = read(root / "evaluation/metrics.json")
assert len(m["rows"]) == 108 and len(m["bindings"]) == 108 and len(m["checks"]) == 108
assert m["finite"] and m["protected_exact"]
for path, digest in m["bindings"].items():
    assert sha(Path(path)) == digest
a = read(root / "replay-audit.json")
assert a["outputs_exact"] == 36 and a["metrics_exact"]
manifest = read(root / "evaluation/render-manifest.json")
review = read(root / "evaluation/visual-review.json")
assert review["reviewed"] and len(manifest["records"]) == len(review["files"]) == 3
for r in manifest["records"]:
    assert sha(root / "evaluation" / r["file"]) == r["sha256"] == review["files"][r["file"]]
failures = dict(Counter(c["method"] for c in m["checks"] if not c["passed"]))
assert failures.get("control", 0) == 27
result = dict(
    stage="complete_anchored_endpoint_diagnostic",
    adoption=False,
    deployable=False,
    numerical_commit=read(root / "source.json")["commit"],
    steps_per_arm=STEPS,
    parameters=130628,
    methods=list(METHODS),
    failures={x: failures.get(x, 0) for x in ("control", *METHODS)},
    raw_ema_exports_exact=True,
    full_state_restore_exact=True,
    original_outputs_exact=36,
    final_images_reviewed=3,
    evaluated_blocks=108,
    samples_per_group=64,
    new_teacher_shapes=False,
    confirmation=False,
    sealed_confirmation_generated=False,
    evidence={
        n: sha(root / n)
        for n in (
            "source.json",
            "plan.json",
            "test-results.json",
            "initial-audit.json",
            "control-preflight.json",
            "prefix-audit.json",
            "snapshot-audit.json",
            "training/sampling.jsonl",
            "replay-audit.json",
            "evaluation/metrics.json",
            "evaluation/visual-review.json",
        )
    },
)
write(root / "final-audit.json", result)
write(root / "workflow-status.json", result)
print({k: v for k, v in result.items() if k != "evidence"})
