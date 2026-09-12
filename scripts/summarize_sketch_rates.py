"""Report paired recursion and augmentation-rate effects with source-cluster intervals."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from run_sketch_rates import METHODS, validate
from summarize_sketch_experiment import clustered_delta, strength_slices, subset
from summarize_training_methods import sha

METRICS = (
    "active_ink_f1",
    "active_mae",
    "active_background_mae",
    "active_background_dark_fraction",
    "profile_width_mae_px",
    "profile_missing_fraction",
    "connection_cut_rate",
    "connection_bridge_rate",
    "mask_boundary_mae",
)


def summaries(report):
    groups = {}
    for row in report["results"]:
        name = subset(row)
        if name is not None:
            groups.setdefault((name, row["method"], row["subnet"]), []).append(row)
    return [
        dict(
            subset=name,
            method=method,
            subnet=subnet,
            cases=len(rows),
            **{
                metric: float(np.mean([r[metric] for r in rows if r.get(metric) is not None]))
                if any(r.get(metric) is not None for r in rows)
                else None
                for metric in METRICS
            },
        )
        for (name, method, subnet), rows in sorted(groups.items())
    ]


def contrast(report, reference, manifest, baseline=None):
    lookup = {(r["case"], r["method"], r["subnet"]): r for r in reference["results"]}
    groups = {}
    for row in report["results"]:
        name = subset(row)
        if name is None or row["method"] == baseline:
            continue
        ref = lookup[row["case"], baseline or row["method"], row["subnet"]]
        case = manifest["cases"][row["case"]]
        cluster = case["source"] if case["kind"] == "real" else (case["kind"], case["variant"])
        for metric in METRICS:
            if row.get(metric) is not None and ref.get(metric) is not None:
                groups.setdefault((name, row["method"], row["subnet"], metric), {}).setdefault(
                    cluster, []
                ).append(row[metric] - ref[metric])
    return [
        dict(
            subset=name,
            method=method,
            subnet=subnet,
            metric=metric,
            baseline=baseline or "same method, dataset recursion",
            **clustered_delta(values),
        )
        for (name, method, subnet, metric), values in sorted(groups.items())
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--through", choices=[0, 2000, 10000], type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "evaluation/cases.json").read_text())
    files = [
        root / "config.toml",
        root / "training-source.json",
        root / "preflight.json",
        root / "evaluation-input-sha256.json",
        root / "evaluation/cases.json",
    ]
    for name, digest in json.loads((root / "evaluation-input-sha256.json").read_text()).items():
        assert sha(root / "evaluation" / name) == digest, name

    def read(path, methods, step, recursion):
        report = json.loads(path.read_text())
        validate(report, methods, step, recursion)
        assert report["manifest_sha256"] == sha(root / "evaluation/cases.json")
        files.append(path)
        return report

    recursion = {}
    for mode in ("raw", "ema"):
        reports = {
            kind: read(
                root / "recursion" / f"{kind}-{mode}/metrics.json",
                ("control", "tone", "blur", "strokes"),
                10000,
                kind,
            )
            for kind in ("dataset", "runtime")
        }
        recursion[mode] = dict(
            summaries={kind: summaries(report) for kind, report in reports.items()},
            paired_runtime_minus_dataset=contrast(reports["runtime"], reports["dataset"], manifest),
            strength_slices={kind: strength_slices(report) for kind, report in reports.items()},
        )
    rates, timings = {}, None
    if args.through:
        initial = read(root / "evaluation/step-0-raw/metrics.json", ("initial",), 0, "runtime")
        rates["initial"] = summaries(initial)
        progress = json.loads((root / "training/progress.json").read_text())
        assert progress["step"] >= args.through and progress["runs"] == list(METHODS)
        files.extend([root / "training/initialization.json", root / "pilot-gate.json"])
        reference = Path(json.loads((root / "training/initialization.json").read_text())["source"])
        with np.load(reference / "weights.npz", allow_pickle=False) as expected:
            for method in METHODS:
                path = root / "training/initial" / method / "weights.npz"
                with np.load(path, allow_pickle=False) as actual:
                    assert set(actual.files) == set(expected.files)
                    for key in actual.files:
                        np.testing.assert_array_equal(actual[key], expected[key])
                files.extend(path.parent.iterdir())
        for step, mode in itertools.product(
            (2000,) if args.through == 2000 else (2000, 10000), ("raw", "ema")
        ):
            report = read(
                root / "evaluation" / f"step-{step}-{mode}/metrics.json", METHODS, step, "runtime"
            )
            assert report["ema"] == (mode == "ema")
            rates[f"{step}-{mode}"] = dict(
                summary=summaries(report),
                paired_minus_p75=contrast(report, report, manifest, "p75"),
                paired_minus_initial=contrast(report, initial, manifest, "initial"),
                strength_slices=strength_slices(report),
            )
            for method in METHODS:
                files.extend((root / "snapshots" / f"step-{step}" / method / mode).iterdir())
        rows = [
            json.loads(line)
            for line in (root / "training/timings.jsonl").read_text().splitlines()
            if json.loads(line)["step"] <= args.through
        ]
        assert [r["step"] for r in rows] == list(range(1, args.through + 1))
        assert all(r["recursion_levels"] == r["scale_level"] + 1 for r in rows)
        assert all(set(r["updates"]) == set(METHODS) for r in rows)
        timing_path = root / f"timings-through-{args.through}.json"
        timing_path.write_text(json.dumps(rows) + "\n")
        files.append(timing_path)
        timings = dict(
            recorded_seconds=sum(r["seconds_before_evaluation"] for r in rows),
            mean_update_seconds={
                m: float(np.mean([r["updates"][m] for r in rows[100:]])) for m in METHODS
            },
        )
        if args.through == 10000:
            for kind, path in (
                ("verification", root / "verification/step-10000-raw-ema.json"),
                ("exports", root / "exports/step-10000/validation.json"),
            ):
                audit = json.loads(path.read_text())
                assert audit["step"] == 10000
                assert {(r["method"], r["mode"]) for r in audit["records"]} == set(
                    itertools.product(METHODS, ("raw", "ema"))
                )
                assert len(audit["records"]) == 6
                for record in audit["records"]:
                    if kind == "verification":
                        assert record["checkpoint_and_optimizer_step"] == 10000
                        assert record["parameters"] == [5933, 10206, 17893]
                        for archive in record["archives"]:
                            for name, digest in archive.items():
                                assert sha(Path(name)) == digest
                                files.append(Path(name))
                    else:
                        assert sha(Path(record["onnx"])) == record["sha256"]
                        files.extend(
                            [Path(record["onnx"]), Path(record["onnx"]).with_suffix(".json")]
                        )
                files.append(path)
            for method in METHODS:
                files.extend(
                    p
                    for p in (root / "training" / method / "checkpoints/10000").rglob("*")
                    if p.is_file()
                )
    result = dict(
        synthetic_only=True,
        through=args.through,
        case_count=1562,
        denoise_steps=1,
        recursion=recursion,
        rates=rates,
        timing=timings,
        artifacts={str(p): sha(p) for p in sorted(set(files))},
        analysis_source_sha256=sha(Path(__file__)),
        limitations="One training seed, development only, synthetic sketches only; "
        "source-cluster intervals conditional on fitted models. Normalized arrays "
        "without BGRA quantization. M/L use internal prefix states in JAX.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
