"""Close this bounded diagnostic after the recorded native-image review."""

import argparse
import json
from collections import Counter
from pathlib import Path

from campaign_protocol import read, sha, write
from cue_capacity import METHODS, SAVED, STEPS, check


def finalize(root):
    check(root)
    tests = read(root / "test-results.json")
    assert tests["passed"]
    for file, digest in tests["logs"].items():
        assert sha(root / file) == digest
    assert read(root / "initial-audit.json")["six_depth_outputs_and_losses_exact"]
    assert read(root / "backend.json")["flags"] == ""
    prefix = read(root / "prefix-audit.json")
    assert prefix["all_state_exact"] and prefix["metrics_exact"]
    for step in SAVED[1:]:
        assert read(root / f"training/restore-{step}.json")["all_state_exact"]
    exports = read(root / "snapshot-audit.json")
    assert exports["raw_ema_exact"] and exports["exports"] == 16
    rows = [json.loads(s) for s in (root / "training/sampling.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == list(range(1, STEPS + 1))
    plan = read(root / "sampling-plan.json")
    for r, p in zip(rows, plan, strict=True):
        assert r["step"] == p["step"] + 1
        assert all(r[k] == p[k] for k in ("indices", "modes", "depth"))
        assert set(r["metrics"]) == set(METHODS)
    assert len({r["input_sha256"] for r in rows}) == STEPS
    out = root / "evaluation"
    metrics = read(out / "metrics.json")
    assert metrics["coverage_exact"] and metrics["protected_exact"]
    assert len(metrics["rows"]) == 2304 and len(metrics["references"]) == 1152
    for file, digest in read(out / "checkpoint-binding.json").items():
        assert sha(Path(file)) == digest
    for file, digest in read(out / "prediction-binding.json").items():
        assert sha(out / file) == digest
    summary = read(out / "summary.json")
    assert summary["oracle_passed"]
    review = read(out / "visual-review.json")
    manifest = read(out / "native-review/manifest.json")
    assert review["reviewed"] and len(review["files"]) == len(manifest["records"]) == 12
    for r in manifest["records"]:
        assert sha(out / "native-review" / r["file"]) == r["sha256"] == review["files"][r["file"]]
        for file, digest in r["bindings"].items():
            assert sha(Path(file)) == digest
    result = dict(
        stage="complete_cue_capacity_diagnostic",
        adoption=False,
        deployable=False,
        numerical_commit=read(root / "source.json")["commit"],
        steps_per_arm=STEPS,
        model_rows=2304,
        reference_rows=1152,
        native_images_reviewed=12,
        all_state_restore_exact=True,
        raw_ema_exports_exact=True,
        depth_counts=dict(Counter(r["depth"] for r in rows)),
        failures={
            f"{s}/{m}": sum(
                not c["passed"] for c in summary["checks"] if c["split"] == s and c["method"] == m
            )
            for s in ("train", "probe")
            for m in METHODS
        },
        evidence={
            n: sha(root / n)
            for n in (
                "source.json",
                "plan.json",
                "data.json",
                "test-results.json",
                "initial-audit.json",
                "prefix-audit.json",
                "snapshot-audit.json",
                "training/sampling.jsonl",
                "evaluation/metrics.json",
                "evaluation/summary.json",
                "evaluation/visual-review.json",
            )
        },
    )
    write(root / "completion.json", result)
    write(root / "workflow-status.json", dict(stage=result["stage"], step=STEPS))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    finalize(p.parse_args().root.resolve())
