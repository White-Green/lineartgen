"""Finish the endpoint-anchor toy after all numerical and visual audits."""

import json
from collections import Counter
from pathlib import Path

from campaign_protocol import read, sha, write
from early_path_fit import METHODS, PARAMETERS, SAVED, START, STEPS, check

root = Path("runs/early-path-20261011").resolve()
plan = check(root)
tests = read(root / "test-results.json")
assert tests["passed"]
for path, digest in tests["logs"].items():
    assert sha(root / path) == digest
a = read(root / "initial-audit.json")
assert (
    a["full_parameters_identical"] and a["parent_full_state_exact"] and a["optimizer_step"] == START
)
a = read(root / "control-preflight.json")
assert (
    a["parent_full_state_exact"] and a["original_batches_exact"] == 6 and a["constant_lr"] == 1e-5
)
a = read(root / "prefix-audit.json")
assert a["all_state_exact"] and a["metrics_exact"] and a["original_control_updates_exact"]
for step in SAVED:
    assert read(root / f"training/restore-audit-{step}.json")["all_state_exact"]
a = read(root / "snapshot-audit.json")
assert a["raw_ema_exact"] and a["exports"] == 16
for path, digest in a["bindings"].items():
    assert sha(Path(path)) == digest
sampling = [json.loads(s) for s in (root / "training/sampling.jsonl").read_text().splitlines()]
assert [r["step"] for r in sampling] == list(range(START + 1, STEPS + 1))
assert all(r["depth"] == (r["step"] - 1) % 3 + 1 and r["label_A"] == 8 for r in sampling)
for name in METHODS:
    assert len({r["inputs_sha256"][name] for r in sampling}) == STEPS - START
assert all(r["focused_early_count"] >= r["control_early_count"] for r in sampling)
exposure = {
    name: sum(r[name] for r in sampling)
    for name in ("zero_count", "focused_count", "control_early_count", "focused_early_count")
}
assert exposure["focused_early_count"] > 10 * exposure["control_early_count"]
write(root / "exposure-audit.json", dict(examples=16 * (STEPS - START), **exposure))
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
assert failures.get("initial", 0) == 5
result = dict(
    stage="complete_early_path_diagnostic",
    adoption=False,
    deployable=False,
    numerical_commit=read(root / "source.json")["commit"],
    start=START,
    final_step=STEPS,
    additional_steps_per_arm=STEPS - START,
    parameters=PARAMETERS,
    methods=list(METHODS),
    failures={x: failures.get(x, 0) for x in ("initial", *METHODS)},
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
            "exposure-audit.json",
            "replay-audit.json",
            "evaluation/metrics.json",
            "evaluation/visual-review.json",
        )
    },
)
write(root / "final-audit.json", result)
write(root / "workflow-status.json", result)
print({k: v for k, v in result.items() if k != "evidence"})
