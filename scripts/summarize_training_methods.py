"""Audit completed measurements and create compact, reproducible comparison evidence.

Bootstrap intervals resample source images or synthetic shapes, not individual
mask/strength conditions. They do not quantify training-seed uncertainty.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from evaluate_training_methods import summarize

METHODS = ("current", "residual", "ot_full", "independent_noise")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(root, name, payload):
    """Retain the measured prefix even when training or provenance later changes."""
    digest = hashlib.sha256(payload).hexdigest()
    path = root / "evidence" / f"{digest}-{name}"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        assert path.read_bytes() == payload, path
    else:
        path.write_bytes(payload)
    return path


def paired_deltas(report, manifest):
    output = []
    selected = {
        "matched_real": lambda r: r["suite"] == "matched_real" and r["strength"] > 0,
        "matched_real_active_lines": lambda r: (
            r["suite"] == "matched_real"
            and r["strength"] > 0
            and r["active_target_ink_fraction"] >= 0.001
        ),
        "geometry": lambda r: r["suite"] == "geometry" and r["strength"] > 0,
    }
    rows = report["results"]
    for subset, select in selected.items():
        metric = "active_ink_f1" if subset == "matched_real_active_lines" else "ink_f1"
        for subnet in range(3):
            baseline = {
                r["case"]: r
                for r in rows
                if select(r)
                and r["method"] == "current"
                and r["subnet"] == subnet
                and r["denoise_steps"] == 5
            }
            for method in METHODS[1:]:
                candidate = {
                    r["case"]: r
                    for r in rows
                    if select(r)
                    and r["method"] == method
                    and r["subnet"] == subnet
                    and r["denoise_steps"] == 5
                }
                assert baseline.keys() == candidate.keys()
                groups = {}
                for case, reference in baseline.items():
                    source = manifest["cases"][case]
                    group = (
                        (source["kind"], source["variant"])
                        if subset == "geometry"
                        else source["source"]
                    )
                    groups.setdefault(group, []).append(candidate[case][metric] - reference[metric])
                totals = np.asarray([sum(values) for values in groups.values()])
                counts = np.asarray([len(values) for values in groups.values()])
                random = np.random.default_rng(20260911)
                indices = random.integers(len(groups), size=(5000, len(groups)))
                samples = totals[indices].sum(axis=1) / counts[indices].sum(axis=1)
                low, high = np.quantile(samples, [0.025, 0.975])
                output.append(
                    dict(
                        subset=subset,
                        metric=metric,
                        method=method,
                        subnet=subnet,
                        denoise_steps=5,
                        cases=len(baseline),
                        groups=len(groups),
                        delta=float(totals.sum() / counts.sum()),
                        clustered_95_interval=[float(low), float(high)],
                    )
                )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--through", type=int, choices=[2000, 10000], default=10000)
    args = parser.parse_args()
    root = args.root
    provenance = freeze(root, "training-source.json", (root / "training-source.json").read_bytes())
    files = [
        provenance,
        root / "evaluation/cases.json",
        root / "noise-assignment.json",
    ]
    manifest = read(root / "evaluation/cases.json")
    assert len(manifest["cases"]) == 434
    for path, digest in read(root / "evaluation-input-sha256.json").items():
        assert sha(root / path) == digest, path
    progress = read(root / "training/progress.json")
    assert progress["step"] >= args.through and progress["runs"] == list(METHODS)
    evaluations = {}
    expected = {
        (case, method, subnet, steps)
        for case in range(434)
        for method in METHODS
        for subnet in range(3)
        for steps in (1, 2, 5)
    }
    for step in (2000,) if args.through == 2000 else (2000, 10000):
        for mode in ("raw", "ema"):
            path = root / "evaluation" / f"step-{step}-{mode}" / "metrics.json"
            report = read(path)
            assert report["checkpoint_step"] == step and report["ema"] == (mode == "ema")
            assert report["cases"] == 434 and report["manifest_sha256"] == sha(
                root / "evaluation/cases.json"
            )
            rows = report["results"]
            assert len(rows) == len(expected)
            assert {
                (r["case"], r["method"], r["subnet"], r["denoise_steps"]) for r in rows
            } == expected
            for row in rows:
                case = manifest["cases"][row["case"]]
                for field in ("suite", "kind", "strength", "partial", "scale_level"):
                    assert row[field] == case[field]
                if row["method"] == "residual":
                    assert row["protected_max_error"] in (None, 0)
            assert summarize(rows) == report["summary"]
            evaluations[f"{step}-{mode}"] = dict(
                summary=report["summary"], paired_deltas=paired_deltas(report, manifest)
            )
            files.append(path)
    # Read only the completed prefix. A later update may currently be appending a line.
    timing_lines = (root / "training/timings.jsonl").read_text().splitlines()[: args.through]
    timings = [json.loads(line) for line in timing_lines]
    assert [r["step"] for r in timings] == list(range(1, args.through + 1))
    assert all(set(r["updates"]) == set(METHODS) for r in timings)
    files.append(
        freeze(
            root, f"timings-through-{args.through}.jsonl", ("\n".join(timing_lines) + "\n").encode()
        )
    )
    steady = [r for r in timings if r["step"] > 100]
    speed = {
        method: dict(
            mean_seconds=float(np.mean([r["updates"][method] for r in steady])),
            median_seconds=float(np.median([r["updates"][method] for r in steady])),
            recorded_update_seconds=sum(r["updates"][method] for r in timings),
        )
        for method in METHODS
    }
    speed.update(
        input_mean_seconds=float(np.mean([r["input_seconds"] for r in steady])),
        preparation_mean_seconds=float(np.mean([r["preparation_seconds"] for r in steady])),
        recorded_seconds_before_evaluation=sum(r["seconds_before_evaluation"] for r in timings),
    )
    final_audit = None
    if args.through == 10000:
        path = root / "verification/step-10000-raw-ema.json"
        verified = read(path)
        assert len(verified["records"]) == 8
        assert {(r["method"], r["mode"]) for r in verified["records"]} == {
            (method, mode) for method in METHODS for mode in ("raw", "ema")
        }
        assert all(
            r["checkpoint_and_optimizer_step"] == 10000
            and r["parameters"] == [5933, 10206, 17893]
            and len(r["comparisons"]) == 6
            for r in verified["records"]
        )
        assert verified["step"] == 10000
        for record in verified["records"]:
            assert {(r["levels"], r["subnet"]) for r in record["comparisons"]} == {
                (levels, subnet) for levels in (1, 6) for subnet in range(3)
            }
            assert len(record["archives"]) == 3
            for archive in record["archives"]:
                assert {Path(p).name for p in archive} == {
                    "config.json",
                    "weights.npz",
                    "metadata.json",
                }
                for path, digest in archive.items():
                    assert sha(Path(path)) == digest, path
        exports = read(root / "exports/step-10000/validation.json")
        assert exports["step"] == 10000 and len(exports["records"]) == 8
        assert {(r["method"], r["mode"]) for r in exports["records"]} == {
            (method, mode) for method in METHODS for mode in ("raw", "ema")
        }
        for record in exports["records"]:
            assert sha(Path(record["onnx"])) == record["sha256"]
        final_audit = dict(trained_verification=verified, exports=exports)
        files += [path, root / "exports/step-10000/validation.json"]
    evidence = dict(
        format_version=1,
        through_step=args.through,
        training_examples_per_method=args.through * 16,
        provenance=read(provenance),
        packaging_source_sha256=sha(Path(__file__)),
        dataset=manifest["dataset"],
        fixed_case_count=434,
        evaluation_input_sha256=read(root / "evaluation-input-sha256.json"),
        evaluations=evaluations,
        timing=speed,
        noise_assignment=read(root / "noise-assignment.json"),
        final_audit=final_audit,
        artifact_sha256={str(path): sha(path) for path in files},
        limitations=(
            "One training seed. Bootstrap intervals are conditional on the fitted models and "
            "resample development source images or synthetic shapes. Timing includes concurrent "
            "preflight work and excludes unrecorded initialization, evaluation and saving "
            "intervals; it is not an isolated hardware benchmark."
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
