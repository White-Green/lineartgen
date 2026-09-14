"""Compare procedural clean target replacement, with a fixed synthetic-sketch recipe."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from run_sketch_rates import validate
from summarize_training_methods import sha

METHODS = ("control", "geometry20")


def pilot_gate(reports, initial, methods=METHODS):
    for report in reports:
        validate(report, methods, 2000, "runtime")
    reference = {(r["subset"], r["subnet"]): r for r in initial["summary"]}
    failures = []
    for row in reports[0]["summary"]:
        if row["subset"] in ("matched_real", "geometry"):
            threshold = min(0.3, reference[row["subset"], row["subnet"]]["ink_f1"] * 0.5)
            if row["ink_f1"] < threshold:
                failures.append(f"Collapse: {row['method']}/{row['subnet']}/{row['subset']}")
    return dict(
        passed=not failures,
        failures=failures,
        criteria="Complete finite evaluations, exact protected pixels, raw clean F1 "
        ">= min(.30, half initial F1). Execution/collapse gate, not quality acceptance.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, default=Path("runs/sketch-rates-20260912"))
    parser.add_argument(
        "--study", choices=["geometry_data", "boundary_loss"], default="geometry_data"
    )
    args = parser.parse_args()
    methods = METHODS if args.study == "geometry_data" else ("control", "boundary")
    root, reference = args.root.resolve(), args.reference_root.resolve()
    config, cases, group = root / "config.toml", root / "evaluation", root / "training"
    initial_method = "p50" if args.study == "geometry_data" else "geometry20"
    initial = reference / "snapshots/step-10000" / initial_method / "raw"

    def status(stage, **extra):
        record = dict(
            stage=stage,
            utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            pid=os.getpid(),
            **extra,
        )
        temporary = root / "workflow-status.partial.json"
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(root / "workflow-status.json")
        print(json.dumps(record), flush=True)

    def run(script, arguments, label, cpu=False):
        status(label)
        env = dict(os.environ)
        if cpu:
            env.update(JAX_PLATFORMS="cpu", OPENBLAS_NUM_THREADS="1")
        with (root / f"{label}.log").open("a") as log:
            subprocess.run(
                [sys.executable, "-u", f"scripts/{script}.py", *map(str, arguments)],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                check=True,
            )

    def evaluate(step, mode):
        folder = cases / f"step-{step}-{mode}"
        path = folder / "metrics.json"
        if not path.exists():
            run(
                "evaluate_training_methods",
                [
                    "--study",
                    args.study,
                    "--config",
                    config,
                    "--cases",
                    cases,
                    "--step",
                    step,
                    "--steps",
                    1,
                    "--recursion",
                    "runtime",
                    "--noise-generator",
                    "numpy",
                    "--output",
                    folder,
                    *(["--initialize-from", initial] if step == 0 else ["--run-group", group]),
                    *(["--raw"] if mode == "raw" else []),
                ],
                f"evaluation-{step}-{mode}",
            )
        report = json.loads(path.read_text())
        validate(report, ("initial",) if step == 0 else methods, step, "runtime")
        assert report["manifest_sha256"] == sha(cases / "cases.json")
        assert report["ema"] == (mode == "ema")
        return report

    try:
        for path, digest in json.loads((root / "training-source.json").read_text())[
            "file_sha256"
        ].items():
            assert sha(Path(path)) == digest, path
        for path, digest in json.loads((root / "reference-model-sha256.json").read_text()).items():
            assert sha(Path(path)) == digest, path
        if not cases.exists():
            cases.mkdir()
            manifest = json.loads((reference / "evaluation/cases.json").read_text())
            files = ["cases.json", *(r["file"] for r in manifest["cases"])]
            for name in files:
                shutil.copyfile(reference / "evaluation" / name, cases / name)
            (root / "evaluation-input-sha256.json").write_text(
                json.dumps({name: sha(cases / name) for name in files}, indent=2) + "\n"
            )
        for name, digest in json.loads((root / "evaluation-input-sha256.json").read_text()).items():
            assert sha(cases / name) == digest, name
        if args.study == "boundary_loss":
            from lineartgen.config import load_config

            calibration = json.loads((root / "calibration.json").read_text())
            assert calibration["coefficient_frozen_before_training"]
            assert load_config(config).loss.boundary_weight == calibration["coefficient"]
            assert sha(initial / "weights.npz") == calibration["reference_weights_sha256"]
        initial_report = evaluate(0, "raw")
        for step in (2000, 10000):
            if (root / "stop").exists():
                status("stop_requested")
                return
            if step == 10000 and not json.loads((root / "pilot-gate.json").read_text())["passed"]:
                status("pilot_needs_review")
                return
            progress_path = group / "progress.json"
            progress = (
                json.loads(progress_path.read_text())["step"] if progress_path.exists() else 0
            )
            if progress < step:
                run(
                    "compare_nested_training",
                    [
                        "--study",
                        args.study,
                        "--config",
                        config,
                        "--run-dir",
                        group,
                        "--stop-file",
                        root / "stop",
                        "--stop-after",
                        step - progress,
                        *(["--resume"] if progress else ["--initialize-from", initial]),
                    ],
                    f"training-{step}",
                )
            if (root / "stop").exists():
                status("stop_requested")
                return
            reports = [evaluate(step, mode) for mode in ("raw", "ema")]
            if step == 2000:
                gate = pilot_gate(reports, initial_report, methods)
                (root / "pilot-gate.json").write_text(json.dumps(gate, indent=2) + "\n")
                status("pilot_evaluated", gate=gate)
        for cpu in (False, True):
            run(
                "verify_training_methods",
                [
                    "--study",
                    args.study,
                    "--root",
                    root,
                    "--exports-only" if cpu else "--check-checkpoints",
                ],
                "exports-10000" if cpu else "verification-10000",
                cpu=cpu,
            )
        status("complete", optimizer_steps=10000, evaluated_steps=[0, 2000, 10000])
    except Exception as error:
        status("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
