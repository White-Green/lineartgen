"""Report procedural training target effects with source-cluster intervals."""

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
from run_geometry_data import METHODS
from run_sketch_rates import validate
from summarize_sketch_experiment import strength_slices
from summarize_sketch_rates import contrast, summaries
from summarize_training_methods import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--through", choices=[2000, 10000], type=int, default=10000)
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
    for name in (
        "reference-model-sha256.json",
        "runtime-parity.json",
        "tests.log",
        "analysis-source.json",
    ):
        if (root / name).exists():
            files.append(root / name)
    for path, digest in json.loads((root / "reference-model-sha256.json").read_text()).items():
        assert sha(Path(path)) == digest, path
    for name, digest in json.loads((root / "evaluation-input-sha256.json").read_text()).items():
        assert sha(root / "evaluation" / name) == digest, name

    def read(path, methods, step, recursion):
        report = json.loads(path.read_text())
        validate(report, methods, step, recursion)
        assert report["manifest_sha256"] == sha(root / "evaluation/cases.json")
        files.append(path)
        return report

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
                paired_minus_control=contrast(report, report, manifest, "control"),
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
        exposures = {}
        for method in METHODS:
            generated = [item for row in rows for item in row["generated"][method]]
            exposures[method] = dict(
                total=args.through * 16,
                real=args.through * 16 - len(generated),
                generated=len(generated),
                blank=sum(r["kinds"] == ["blank"] for r in generated),
                kind_groups=dict(Counter(k for r in generated for k in r["kinds"])),
            )
        assert exposures["control"]["generated"] == 0
        assert 0.17 < exposures["geometry20"]["generated"] / (args.through * 16) < 0.23
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
                assert len(audit["records"]) == 4
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
        data_comparison=rates,
        exposures=exposures,
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
