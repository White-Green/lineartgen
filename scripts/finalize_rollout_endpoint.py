"""Audit both inference counts after reviewing all ten fixed geometry sheets."""

import json
from pathlib import Path

import numpy as np
from campaign_protocol import read, sha, write
from rollout_endpoint_fit import METHODS, PROBE, SAVED, START, STEPS, TRAIN, check

root = Path("runs/rollout-endpoint-20261011").resolve()
plan = check(root)
tests = read(root / "test-results.json")
assert tests["passed"]
for name, digest in tests["logs"].items():
    assert sha(root / name) == digest
initial = read(root / "initial-audit.json")
data = read(root / "data-audit.json")
assert data["disjoint"] and data["train"] == list(TRAIN) and data["probe"] == list(PROBE)
assert data["inputs_sha256"] == sha(root / "inputs.npz")
assert len(data["rows"]) == 8 and all(r["antialias_pixels"] > 0 for r in data["rows"])
assert (
    initial["initial_parameters_exact"]
    and initial["optimizer_step"] == START
    and initial["parent_full_state_exact"]
)
assert read(root / "prefix-audit.json")["all_state_exact"]
assert read(root / "prefix-audit.json")["metrics_exact"]
assert read(root / "prefix-audit.json")["depths"] == list(range(1, 7))
for step in SAVED:
    assert read(root / f"training/restore-audit-{step}.json")["all_state_exact"]
audit = read(root / "snapshot-audit.json")
assert audit["exports"] == 16 and audit["raw_ema_exact"]
for path, digest in audit["bindings"].items():
    assert sha(Path(path)) == digest
all_metrics = {}
for steps, directory in ((2, "evaluation"), (32, "evaluation32")):
    folder = root / directory
    metrics = read(folder / "metrics.json")
    assert len(metrics["rows"]) == 288 and metrics["finite"] and metrics["protected_exact"]
    expected = {
        (m, i, d, n) for m in METHODS for i in TRAIN + PROBE for d in (1, 3, 6) for n in range(4)
    }
    assert {(r["method"], r["case"], r["depth"], r["draw"]) for r in metrics["rows"]} == expected
    assert len(metrics["checks"]) == 264 and len(metrics["retention_checks"]) == 192
    for m in METHODS:
        selected = [c for c in metrics["checks"] if c["method"] == m]
        assert len(selected) == 88 and metrics["fit_passed"][m] == all(
            c["passed"] for c in selected
        )
    assert metrics["retention_passed"] == all(c["passed"] for c in metrics["retention_checks"])
    assert len(metrics["bindings"]) == 75
    for path, digest in metrics["bindings"].items():
        assert sha(Path(path)) == digest
        with np.load(path) as arrays:
            assert arrays["prediction"].shape == arrays["physical"].shape == (4, 512, 512, 1)
            assert np.isfinite(arrays["physical"]).all()
            np.testing.assert_array_equal(arrays["prediction"], np.clip(arrays["physical"], -1, 1))
    review = read(folder / "visual-review.json")
    render = read(folder / "render-manifest.json")
    assert review["reviewed"] and len(review["files"]) == len(render["records"]) == 5
    assert render["all_four_fixed_draws"] and render["crops_from_AD"]
    for r in render["records"]:
        assert sha(folder / r["file"]) == r["sha256"] == review["files"][r["file"]]
    replay = read(root / f"replay-audit-{steps}.json")
    assert replay["exact"] and len(replay["files"]) == len(set(replay["files"])) == (
        25 if steps == 32 else 0
    )
    assert replay["metric_rows_exact"] == (96 if steps == 32 else 0)
    all_metrics[steps] = metrics
assert read(root / "control-preflight.json")["eight_CA_outputs_exact"]
sampling = [json.loads(s) for s in (root / "training/sampling.jsonl").read_text().splitlines()]
assert [r["step"] for r in sampling] == list(range(START + 1, STEPS + 1))
assert all(r["depth"] == (r["step"] - 1) % 6 + 1 for r in sampling)
assert len({r["inputs_sha256"] for r in sampling}) == STEPS - START
assert sum(not c["passed"] for c in all_metrics[32]["checks"] if c["method"] == "initial") == 61
result = dict(
    stage="complete_rollout_endpoint_diagnostic",
    adoption=False,
    deployable=False,
    numerical_commit=read(root / "source.json")["commit"],
    start=START,
    final_step=STEPS,
    additional_steps_per_arm=STEPS - START,
    frozen_initial_replay_exact=True,
    parameters=134688,
    trained_depths=list(range(1, 7)),
    fixed_train_pairs=4,
    unfitted_probe_pairs=4,
    primary_inference_steps=2,
    evaluations={
        steps: dict(
            fit_failures={
                m: sum(not c["passed"] for c in metrics["checks"] if c["method"] == m)
                for m in METHODS
            },
            retention_failures={
                f"{a}_vs_{b}": sum(
                    not c["passed"]
                    for c in metrics["retention_checks"]
                    if c["method"] == a and c["reference"] == b
                )
                for a, b in (("teacher", "initial"), ("rollout", "initial"), ("rollout", "teacher"))
            },
        )
        for steps, metrics in all_metrics.items()
    },
    final_images_reviewed=10,
    full_state_restore_exact=True,
    raw_ema_exports_exact=True,
    finite=True,
    protected_pixels_exact=True,
    confirmation=False,
    sealed_confirmation_generated=False,
    evidence={
        name: sha(root / name)
        for name in (
            "source.json",
            "plan.json",
            "test-results.json",
            "target-oracle.json",
            "data-audit.json",
            "initial-audit.json",
            "control-preflight.json",
            "training/sampling.jsonl",
            "prefix-audit.json",
            "replay-audit-2.json",
            "replay-audit-32.json",
            "snapshot-audit.json",
            "evaluation/metrics.json",
            "evaluation/visual-review.json",
            "evaluation/render-manifest.json",
            "evaluation32/metrics.json",
            "evaluation32/visual-review.json",
            "evaluation32/render-manifest.json",
        )
    },
)
write(root / "final-audit.json", result)
write(root / "workflow-status.json", result)
print({k: v for k, v in result.items() if k != "evidence"})
