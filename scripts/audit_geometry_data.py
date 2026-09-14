"""Verify the completed D1 evidence and package runtime/illustration diagnostics."""

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
from evaluate_training_methods import summarize
from run_geometry_data import METHODS
from run_sketch_rates import validate
from summarize_sketch_experiment import subset
from summarize_training_methods import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument(
        "--study", choices=["geometry_data", "boundary_loss"], default="geometry_data"
    )
    args = parser.parse_args()
    methods = METHODS if args.study == "geometry_data" else ("control", "boundary")
    root = args.root.resolve()
    benchmark = json.loads(args.benchmark.read_text())
    assert benchmark["through"] == 10000
    files = {Path(p) for p in benchmark["artifacts"]}
    for path, digest in benchmark["artifacts"].items():
        assert sha(Path(path)) == digest, path
    files.add(args.benchmark.resolve())
    source = json.loads((root / "training-source.json").read_text())
    for path, digest in source["file_sha256"].items():
        assert sha(Path(path)) == digest, path
    assert sha(Path(source["archive"])) == source["archive_sha256"]
    files.add(Path(source["archive"]).resolve())
    manifest = json.loads((root / "evaluation/cases.json").read_text())
    images = json.loads((root / "evaluation-input-sha256.json").read_text())
    assert len(images) == 1563
    for name, digest in images.items():
        assert sha(root / "evaluation" / name) == digest, name
    expected_outputs, details = 0, []
    for step, mode in [(0, "raw"), *itertools.product((2000, 10000), ("raw", "ema"))]:
        report = json.loads((root / "evaluation" / f"step-{step}-{mode}/metrics.json").read_text())
        validate(report, ("initial",) if step == 0 else methods, step, "runtime")
        assert report["summary"] == summarize(report["results"])
        expected_outputs += len(report["results"])
        groups = {}
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
            if step == 10000 and subset(row) is not None:
                groups.setdefault(
                    (subset(row), row["kind"], row["method"], row["subnet"]), []
                ).append(row)
        for key, rows in sorted(groups.items()):
            values = {}
            for metric in ("active_ink_f1", "profile_offset_mae_px", "profile_offset_variation_px"):
                present = [r[metric] for r in rows if r.get(metric) is not None]
                values[metric] = float(np.mean(present)) if present else None
            details.append(
                dict(
                    mode=mode,
                    subset=key[0],
                    kind=key[1],
                    method=key[2],
                    subnet=key[3],
                    cases=len(rows),
                    **values,
                )
            )
    assert expected_outputs == 42174
    timings = [json.loads(x) for x in (root / "training/timings.jsonl").read_text().splitlines()]
    assert [r["step"] for r in timings] == list(range(1, 10001))
    from lineartgen.config import load_config

    config = load_config(root / "config.toml")
    # Reconstruct only the target-selection RNG, independently of the expensive rasterizer.
    for row in timings:
        step = row["step"] - 1
        rng = np.random.default_rng(np.random.SeedSequence([config.train.seed, step, 0x47454F4D]))
        selected = np.flatnonzero(rng.random(16) < 0.2).tolist()
        actual = row["generated"]["geometry20" if args.study == "geometry_data" else "boundary"]
        if args.study == "geometry_data":
            assert not row["generated"]["control"]
        else:
            assert row["generated"]["control"] == actual
        assert [r["item"] for r in actual] == selected
        assert all(r["key"] == [config.train.seed, step, r["item"], 0x50415448] for r in actual)
    artifacts = {}
    illustrations = []
    for method, mode in itertools.product(methods, ("raw", "ema")):
        folder = root / "evaluation" / f"widths-10000-{method}-{mode}"
        record = json.loads((folder / "verification.json").read_text())
        assert [r["case"] for r in record["records"]] == [256, 479, 480, 1376]
        assert record["weights_sha256"] == sha(
            root / "snapshots/step-10000" / method / mode / "weights.npz"
        )
        assert record["metrics_sha256"] == sha(
            root / "evaluation" / f"step-10000-{mode}/metrics.json"
        )
        for row in record["records"]:
            assert row["recursion_levels"] == 3 and sha(Path(row["image"])) == row["sha256"]
            files.add(Path(row["image"]))
        files.add(folder / "verification.json")
        illustrations.append(record)
    for name, suffix in [
        ("runtime-parity.json", "-runtime.json"),
        ("runtime-bgra.json", "-bgra.json"),
    ]:
        path = root / name
        audit = json.loads(path.read_text())
        assert len(audit["records"]) == 16
        if name == "runtime-parity.json":
            assert max(r["max_abs_error"] for r in audit["records"]) < 2e-5
            assert all(r["protected_exact"] and r["levels"] == 3 for r in audit["records"])
        else:
            assert all(
                r["protected_bgra_exact"] and r["identity_exact"] and r["deterministic"]
                for r in audit["records"]
            )
        destination = args.benchmark.with_name(args.benchmark.stem + suffix)
        destination.write_text(json.dumps(audit, indent=2) + "\n")
        files.update((path, destination.resolve()))
    for suffix, result in [
        ("-illustrations.json", dict(records=illustrations)),
        (
            "-diagnostics.json",
            dict(
                records=details,
                primary_outputs=expected_outputs,
                recursion_counts=dict(Counter(r["recursion_levels"] for r in timings)),
                selection_rng_replay_exact=True,
                note="Offset metrics are conditional on visible profiles; "
                "variation is not an exact curvature metric.",
            ),
        ),
    ]:
        destination = args.benchmark.with_name(args.benchmark.stem + suffix)
        destination.write_text(json.dumps(result, indent=2) + "\n")
        files.add(destination.resolve())
    if (root / "generator-preview").exists():
        files.update(p for p in (root / "generator-preview").iterdir() if p.is_file())
    files.update(
        (root / name)
        for name in ["initial-reevaluation.json", "workflow-status.json", "finish-status.json"]
    )
    for path in sorted(files):
        artifacts[str(path)] = sha(path)
    destination = args.benchmark.with_name(args.benchmark.stem + "-artifacts.json")
    destination.write_text(
        json.dumps(dict(artifacts=artifacts, source_sha256=sha(Path(__file__))), indent=2) + "\n"
    )
    print(
        json.dumps(
            dict(
                primary_outputs=expected_outputs,
                artifacts=len(artifacts),
                artifact_manifest=str(destination),
            )
        )
    )


if __name__ == "__main__":
    main()
