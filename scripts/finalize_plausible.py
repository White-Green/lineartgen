"""Close the generator pilot after numerical audit and native image review."""

import argparse
import json
from collections import Counter
from pathlib import Path

from campaign_protocol import read, sha, write
from plausible_study import METHODS, SAVED, STEPS, check


def finalize(root):
    check(root)
    tests = read(root / "test-results.json")
    assert tests["passed"]
    for file, h in tests["logs"].items():
        assert sha(root / file) == h
    spec = read(root / "metric-specification.json")
    assert spec["passed"]
    for file, h in spec["files"].items():
        assert sha(root / file) == h
    assert read(root / "initial-audit.json")["six_depth_outputs_exact"]
    assert read(root / "backend.json")["flags"] == ""
    prefix = read(root / "prefix-audit.json")
    assert prefix["full_state_exact"] and prefix["metrics_exact"]
    for step in SAVED[1:]:
        assert read(root / f"training/restore-{step}.json")["full_state_exact"]
    snapshots = read(root / "snapshot-audit.json")
    assert snapshots["raw_ema_exact"] and snapshots["exports"] == 24
    for p, h in snapshots["bindings"].items():
        assert sha(Path(p)) == h
    sampling = [json.loads(s) for s in (root / "training/sampling.jsonl").read_text().splitlines()]
    plan = read(root / "sampling-plan.json")
    assert [r["step"] for r in sampling] == list(range(1, STEPS + 1))
    for r, p in zip(sampling, plan, strict=True):
        assert all(r[k] == p[k] for k in ("indices", "choices", "modes", "depth"))
        assert r["step"] == p["step"] + 1 and set(r["metrics"]) == set(METHODS)
    assert len({r["input_sha256"] for r in sampling}) == STEPS
    out = root / "evaluation"
    report = read(out / "metrics.json")
    assert report["coverage_exact"] and report["protected_exact"]
    assert len(report["rows"]) == 3072 and len(report["references"]) == 1920
    for p, h in read(out / "checkpoint-binding.json").items():
        assert sha(Path(p)) == h
    for p, h in read(out / "prediction-binding.json").items():
        assert sha(out / p) == h
    manifest = read(out / "native-review/manifest.json")
    review = read(out / "visual-review.json")
    assert review["reviewed"] and len(review["files"]) == len(manifest["records"]) == 12
    for r in manifest["records"]:
        assert sha(out / "native-review" / r["file"]) == r["sha256"] == review["files"][r["file"]]
        for p, h in r["bindings"].items():
            assert sha(Path(p)) == h
    result = dict(
        stage="complete_plausible_lineart_pilot",
        adoption=False,
        deployable=False,
        numerical_commit=read(root / "source.json")["commit"],
        steps_per_arm=STEPS,
        models=METHODS,
        depth_counts=dict(Counter(r["depth"] for r in sampling)),
        model_rows=3072,
        reference_rows=1920,
        native_images_reviewed=12,
        full_state_exact=True,
        raw_ema_exports_exact=True,
        evidence={
            n: sha(root / n)
            for n in (
                "source.json",
                "plan.json",
                "data.json",
                "test-results.json",
                "metric-specification.json",
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
