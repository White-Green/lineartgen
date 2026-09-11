"""Run shared initialization, 2k pilot and 10k total adaptation with fixed synthetic inputs.

Resume only after the previous process has exited. A stale log is never treated as
evidence that a live process has stopped. The stop file is honored between complete updates.
"""

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

METHODS = ("control", "tone", "blur", "strokes")


def validate_report(report, step, *, initial=False):
    methods = ("initial",) if initial else METHODS
    expected = set(itertools.product(range(1562), methods, range(3), (1, 5)))
    actual = [(r["case"], r["method"], r["subnet"], r["denoise_steps"]) for r in report["results"]]
    if report["checkpoint_step"] != step or report["cases"] != 1562:
        raise ValueError("Wrong checkpoint or incomplete cases")
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("Missing or duplicate case/method/width/iteration combinations")
    for row in report["results"]:
        if row["protected_max_error"] not in (None, 0):
            raise ValueError("Protected input changed")
        if any(isinstance(v, float) and not math.isfinite(v) for v in row.values()):
            raise ValueError("Nonfinite evaluation metric")


def pilot_gate(reports):
    failures = []
    for report in reports:
        try:
            validate_report(report, 2000)
        except ValueError as error:
            failures.append(str(error))
    selected = [
        r
        for r in reports[0]["summary"]
        if r["subset"] in ("matched_real", "geometry") and r["denoise_steps"] == 1
    ]
    if len(selected) != 24:
        failures.append("Missing clean-input summaries")
    for row in selected:
        if row["ink_f1"] < 0.3:
            failures.append(f"Clean-input collapse: {row['method']}/{row['subnet']}")
    return dict(
        passed=not failures,
        failures=failures,
        criteria="Complete finite raw/EMA evaluations; exact protected pixels; "
        "raw clean matched-real and geometry F1 >= .30 at one denoising step. "
        "Execution/collapse gate only, no early ranking rejection.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--initialize-from", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config, group, cases = root / "config.toml", root / "training", root / "evaluation"
    stop = root / "stop"

    def status(stage, **extra):
        record = dict(stage=stage, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra)
        temporary = root / "workflow-status.partial.json"
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(root / "workflow-status.json")
        print(json.dumps(record), flush=True)

    def run(script, arguments, label, *, cpu=False):
        status(label)
        env = os.environ.copy()
        if cpu:
            env.update(JAX_PLATFORMS="cpu", OPENBLAS_NUM_THREADS="1")
        with (root / f"{label}.log").open("a") as output:
            subprocess.run(
                [sys.executable, "-u", f"scripts/{script}.py", *map(str, arguments)],
                stdout=output,
                stderr=subprocess.STDOUT,
                env=env,
                check=True,
            )

    def progress():
        path = group / "progress.json"
        return json.loads(path.read_text())["step"] if path.is_file() else 0

    def evaluate(step, mode, initial=False):
        destination = cases / f"step-{step}-{mode}" / "metrics.json"
        if not destination.exists():
            flags = (
                ["--initialize-from", args.initialize_from] if initial else ["--run-group", group]
            )
            if mode == "raw":
                flags.append("--raw")
            run(
                "evaluate_training_methods",
                [
                    "--study",
                    "sketch",
                    "--config",
                    config,
                    "--cases",
                    cases,
                    "--step",
                    step,
                    "--steps",
                    1,
                    5,
                    *flags,
                ],
                f"evaluation-{step}-{mode}",
            )
        report = json.loads(destination.read_text())
        validate_report(report, step, initial=initial)
        return report

    common = ["--study", "sketch", "--config", config, "--run-dir", group, "--stop-file", stop]
    try:
        evaluate(0, "raw", initial=True)
        for step in (2000, 10000):
            if stop.exists():
                status("stop_requested")
                return
            if step == 10000 and not json.loads((root / "pilot-gate.json").read_text())["passed"]:
                status("pilot_needs_review")
                return
            if progress() < step:
                flags = ["--resume"] if progress() else ["--initialize-from", args.initialize_from]
                run(
                    "compare_nested_training",
                    [*common, *flags, "--stop-after", step - progress()],
                    f"training-{step}",
                )
            if stop.exists():
                status("stop_requested")
                return
            reports = [evaluate(step, mode) for mode in ("raw", "ema")]
            if step == 2000:
                gate = pilot_gate(reports)
                (root / "pilot-gate.json").write_text(json.dumps(gate, indent=2) + "\n")
                status("pilot_evaluated", gate=gate)
        run(
            "verify_training_methods",
            ["--study", "sketch", "--root", root, "--check-checkpoints"],
            "verification-10000",
        )
        run(
            "verify_training_methods",
            ["--study", "sketch", "--root", root, "--exports-only"],
            "exports-10000",
            cpu=True,
        )
        status("complete", optimizer_steps=progress(), evaluated_steps=[0, 2000, 10000])
    except Exception as error:
        status("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
