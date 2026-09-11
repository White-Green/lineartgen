"""Audit synthetic sketch comparisons and paired source-cluster uncertainty.

Intervals describe variation over the fixed development sources, conditional on
one training seed. They do not estimate generalization to real rough sketches.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from evaluate_training_methods import summarize
from run_sketch_experiment import METHODS, validate_report
from summarize_training_methods import freeze, read, sha


def subset(row):
    if row["strength"] <= 0:
        return None
    if row["suite"] == "matched_real":
        return "clean_real" if row["active_target_ink_fraction"] >= 0.001 else None
    if row["suite"] == "geometry":
        return "clean_geometry"
    if row["suite"].startswith("sketch_"):
        return f"{row['suite']}_{row['family']}_{row['severity']}"
    return None


def clustered_delta(values):
    """Case-weighted mean, resampling whole source clusters with their case counts."""
    totals = np.asarray([sum(v) for v in values.values()])
    counts = np.asarray([len(v) for v in values.values()])
    if not len(values):
        return None
    random = np.random.default_rng(20260911)
    indices = random.integers(len(values), size=(5000, len(values)))
    samples = totals[indices].sum(axis=1) / counts[indices].sum(axis=1)
    return dict(
        cases=int(counts.sum()),
        groups=len(values),
        delta=float(totals.sum() / counts.sum()),
        clustered_95_interval=np.quantile(samples, [0.025, 0.975]).tolist(),
    )


def paired_deltas(report, initial, manifest):
    rows = report["results"]
    lookup = {
        (r["case"], r["method"], r["subnet"], r["denoise_steps"]): r
        for r in rows + initial["results"]
    }
    groups = {}
    for row in rows:
        name = subset(row)
        if name is not None:
            groups.setdefault(
                (name, row["method"], row["subnet"], row["denoise_steps"]), []
            ).append(row)
    records = []
    for (name, method, subnet, steps), selected in groups.items():
        for baseline in ("control", "initial"):
            if method == baseline:
                continue
            metrics = ["active_ink_f1", "active_mae"]
            if "geometry" in name:
                metrics += ["profile_width_mae_px", "connection_cut_rate", "connection_bridge_rate"]
            for metric in metrics:
                differences = {}
                for row in selected:
                    reference = lookup[row["case"], baseline, subnet, steps]
                    if row.get(metric) is None or reference.get(metric) is None:
                        continue
                    case = manifest["cases"][row["case"]]
                    cluster = (
                        case["source"]
                        if case["kind"] == "real"
                        else (case["kind"], case["variant"])
                    )
                    differences.setdefault(cluster, []).append(row[metric] - reference[metric])
                result = clustered_delta(differences)
                if result is not None:
                    records.append(
                        dict(
                            subset=name,
                            method=method,
                            baseline=baseline,
                            subnet=subnet,
                            denoise_steps=steps,
                            metric=metric,
                            **result,
                        )
                    )
    return records


def strength_slices(report):
    groups = {}
    for row in report["results"]:
        name = subset(row)
        if name is not None:
            key = (
                name,
                row["method"],
                row["subnet"],
                row["denoise_steps"],
                row["strength"],
                row["partial"],
            )
            groups.setdefault(key, []).append(row)
    return [
        dict(
            subset=k[0],
            method=k[1],
            subnet=k[2],
            denoise_steps=k[3],
            strength=k[4],
            partial=k[5],
            cases=len(rows),
            active_ink_f1=float(np.mean([r["active_ink_f1"] for r in rows])),
            active_mae=float(np.mean([r["active_mae"] for r in rows])),
        )
        for k, rows in sorted(groups.items())
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--through", type=int, choices=[2000, 10000], default=10000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = root / "evaluation/cases.json"
    manifest = read(manifest_path)
    assert len(manifest["cases"]) == 1562
    files = [
        manifest_path,
        root / "training-source.json",
        root / "preflight.json",
        root / "training/initialization.json",
        root / "pilot-gate.json",
    ]
    for name, expected in read(root / "evaluation-input-sha256.json").items():
        assert sha(root / "evaluation" / name) == expected, name
    progress = read(root / "training/progress.json")
    assert progress["step"] >= args.through and progress["runs"] == list(METHODS)
    source = Path(read(root / "training/initialization.json")["source"])
    with np.load(source / "weights.npz", allow_pickle=False) as reference:
        for method in METHODS:
            directory = root / "training/initial" / method
            with np.load(directory / "weights.npz", allow_pickle=False) as actual:
                assert set(actual.files) == set(reference.files)
                for name in actual.files:
                    np.testing.assert_array_equal(actual[name], reference[name])
            files.extend(directory.iterdir())
    initial_path = root / "evaluation/step-0-raw/metrics.json"
    initial = read(initial_path)
    validate_report(initial, 0, initial=True)
    assert initial["manifest_sha256"] == sha(manifest_path) and not initial["ema"]
    files.append(initial_path)
    evaluations = {
        "0-raw": dict(summary=initial["summary"], strength_slices=strength_slices(initial))
    }
    steps = (2000,) if args.through == 2000 else (2000, 10000)
    for step, mode in itertools.product(steps, ("raw", "ema")):
        path = root / "evaluation" / f"step-{step}-{mode}" / "metrics.json"
        report = read(path)
        validate_report(report, step)
        assert report["ema"] == (mode == "ema") and report["manifest_sha256"] == sha(manifest_path)
        for row in report["results"]:
            case = manifest["cases"][row["case"]]
            for field in (
                "suite",
                "kind",
                "strength",
                "partial",
                "scale_level",
                "family",
                "severity",
            ):
                assert row.get(field) == case.get(field), (row["case"], field)
            if case["suite"] == "sketch_geometry" and case["partial"]:
                assert row.get("profile_width_mae_px") is None
        assert summarize(report["results"]) == report["summary"]
        evaluations[f"{step}-{mode}"] = dict(
            summary=report["summary"],
            paired_deltas=paired_deltas(report, initial, manifest),
            strength_slices=strength_slices(report),
        )
        files.append(path)
        for method in METHODS:
            files.extend((root / "snapshots" / f"step-{step}" / method / mode).iterdir())
    lines = (root / "training/timings.jsonl").read_text().splitlines()[: args.through]
    timings = [json.loads(line) for line in lines]
    assert [r["step"] for r in timings] == list(range(1, args.through + 1))
    assert all(
        set(r["updates"]) == set(METHODS) and r["recursion_levels"] == r["scale_level"] + 1
        for r in timings
    )
    files.append(
        freeze(root, f"timings-through-{args.through}.jsonl", ("\n".join(lines) + "\n").encode())
    )
    steady = timings[100:]
    speed = {
        m: dict(
            mean_seconds=float(np.mean([r["updates"][m] for r in steady])),
            recorded_update_seconds=sum(r["updates"][m] for r in timings),
        )
        for m in METHODS
    }
    speed.update(
        input_mean_seconds=float(np.mean([r["input_seconds"] for r in steady])),
        preparation_mean_seconds=float(np.mean([r["preparation_seconds"] for r in steady])),
        recorded_seconds_before_evaluation=sum(r["seconds_before_evaluation"] for r in timings),
    )
    audits = None
    if args.through == 10000:
        verification_path = root / "verification/step-10000-raw-ema.json"
        export_path = root / "exports/step-10000/validation.json"
        verification, exports = read(verification_path), read(export_path)
        expected = set(itertools.product(METHODS, ("raw", "ema")))
        for report in (verification, exports):
            assert report["step"] == 10000 and len(report["records"]) == 8
            assert {(r["method"], r["mode"]) for r in report["records"]} == expected
        for record in verification["records"]:
            assert record["checkpoint_and_optimizer_step"] == 10000
            assert record["parameters"] == [5933, 10206, 17893]
            assert {(r["levels"], r["subnet"]) for r in record["comparisons"]} == set(
                itertools.product((1, 6), range(3))
            )
            for archive in record["archives"]:
                for name, digest in archive.items():
                    assert sha(Path(name)) == digest
                    files.append(Path(name))
        for record in exports["records"]:
            assert sha(Path(record["onnx"])) == record["sha256"]
            files.append(Path(record["onnx"]))
            files.append(Path(record["onnx"]).with_suffix(".json"))
        audits = dict(verification=verification, exports=exports)
        files.extend((verification_path, export_path))
        for method in METHODS:
            files.extend(
                p
                for p in (root / "training" / method / "checkpoints/10000").rglob("*")
                if p.is_file()
            )
    evidence = dict(
        format_version=1,
        through_step=args.through,
        synthetic_only=True,
        adaptation_examples_per_method=args.through * 16,
        provenance=read(root / "training-source.json"),
        fixed_case_count=1562,
        dataset=manifest["dataset"],
        evaluations=evaluations,
        timing=speed,
        final_audit=audits,
        evaluation_input_sha256=read(root / "evaluation-input-sha256.json"),
        artifact_sha256={str(p): sha(p) for p in sorted(set(files))},
        packaging_source_sha256=sha(Path(__file__)),
        limitations="One training seed and synthetic rough inputs only. Source-cluster intervals "
        "do not quantify training-seed variation. Strong cases partly exceed training ranges. "
        "Global geometry probes of rough inputs use full edits only. Timing excludes unrecorded "
        "evaluation/initialization/checkpoint intervals and is not an isolated hardware benchmark.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
