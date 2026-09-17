"""Paired recursion diagnosis and three-rate synthetic sketch adaptation.

Run recursion first and inspect it before starting training. Resume only after the
previous process has exited. No completed evaluations or checkpoints are overwritten.
"""

import argparse
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from evaluate_training_methods import summarize
from summarize_training_methods import sha

METHODS = ("p75", "p50", "p25")


def validate(report, methods, step, recursion, *, case_count=1562):
    expected = set(itertools.product(range(case_count), methods, range(3), (1,)))
    actual = [(r["case"], r["method"], r["subnet"], r["denoise_steps"]) for r in report["results"]]
    assert report["checkpoint_step"] == step and report["cases"] == case_count
    assert report["recursion"] == recursion and report["noise_generator"] == "numpy"
    assert len(actual) == len(expected) and set(actual) == expected
    for row in report["results"]:
        assert row["protected_max_error"] in (None, 0)
        assert all(not isinstance(v, float) or math.isfinite(v) for v in row.values())
        assert row["recursion_levels"] == (3 if recursion == "runtime" else row["scale_level"] + 1)


def pilot_gate(reports, initial):
    failures = []
    reference = {(r["subset"], r["subnet"]): r for r in initial["summary"]}
    for report in reports:
        validate(report, METHODS, 2000, "runtime")
    for row in reports[0]["summary"]:
        if row["subset"] not in ("matched_real", "geometry"):
            continue
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
    parser.add_argument("--reference-root", type=Path, default=Path("runs/sketch-20260911"))
    parser.add_argument("--phase", choices=["recursion", "training"], required=True)
    args = parser.parse_args()
    root, reference = args.root.resolve(), args.reference_root.resolve()
    config, cases, group = root / "config.toml", root / "evaluation", root / "training"
    initial = reference / "snapshots/step-10000/strokes/raw"

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

    def evaluate(step, mode, recursion, *, old=False):
        folder = (
            root / "recursion" / f"{recursion}-{mode}" if old else cases / f"step-{step}-{mode}"
        )
        path = folder / "metrics.json"
        if not path.exists():
            run(
                "evaluate_training_methods",
                [
                    "--study",
                    "sketch" if old else "sketch_rates",
                    "--config",
                    reference / "config.toml" if old else config,
                    "--cases",
                    cases,
                    "--run-group",
                    reference / "training" if old else group,
                    "--step",
                    step,
                    "--steps",
                    1,
                    "--recursion",
                    recursion,
                    "--noise-generator",
                    "numpy",
                    "--output",
                    folder,
                    *(["--raw"] if mode == "raw" else []),
                ],
                f"{'recursion-' + recursion if old else 'evaluation-' + str(step)}-{mode}",
            )
        report = json.loads(path.read_text())
        validate(
            report, ("control", "tone", "blur", "strokes") if old else METHODS, step, recursion
        )
        assert report["manifest_sha256"] == sha(cases / "cases.json")
        assert report["ema"] == (mode == "ema")
        return report

    try:
        for path, digest in json.loads((root / "training-source.json").read_text())[
            "file_sha256"
        ].items():
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
        if args.phase == "recursion":
            for mode, recursion in itertools.product(("raw", "ema"), ("dataset", "runtime")):
                evaluate(10000, mode, recursion, old=True)
            status("recursion_complete")
            return
        # Re-use exactly the already evaluated raw strokes warmstart at runtime depth.
        previous = evaluate(10000, "raw", "runtime", old=True)
        initial_path = cases / "step-0-raw/metrics.json"
        if not initial_path.exists():
            initial_report = dict(previous)
            initial_report["results"] = [
                dict(r, method="initial") for r in previous["results"] if r["method"] == "strokes"
            ]
            initial_report.update(
                checkpoint_step=0,
                summary=summarize(initial_report["results"]),
                source_archive=str(initial),
                source_weights_sha256=sha(initial / "weights.npz"),
                reused_from=str(root / "recursion/runtime-raw/metrics.json"),
            )
            initial_path.parent.mkdir()
            initial_path.write_text(json.dumps(initial_report, indent=2, allow_nan=False) + "\n")
        initial_report = json.loads(initial_path.read_text())
        validate(initial_report, ("initial",), 0, "runtime")
        for step in (2000, 10000):
            if (root / "stop").exists():
                status("stop_requested")
                return
            if step == 10000 and not json.loads((root / "pilot-gate.json").read_text())["passed"]:
                status("pilot_needs_review")
                return
            progress = (
                json.loads((group / "progress.json").read_text())["step"] if group.exists() else 0
            )
            if progress < step:
                run(
                    "compare_nested_training",
                    [
                        "--study",
                        "sketch_rates",
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
            reports = [evaluate(step, mode, "runtime") for mode in ("raw", "ema")]
            if step == 2000:
                gate = pilot_gate(reports, initial_report)
                (root / "pilot-gate.json").write_text(json.dumps(gate, indent=2) + "\n")
                status("pilot_evaluated", gate=gate)
        for cpu in (False, True):
            run(
                "verify_training_methods",
                [
                    "--study",
                    "sketch_rates",
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
