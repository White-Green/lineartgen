"""Run the bounded four-method experiment, retaining both pilot and final weights.

The pilot gate checks execution/collapse, not whether a method wins. Do not
discard a method merely for ranking below the control after 2,000 updates.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def pilot_gate(raw, ema):
    failures = []
    for report in (raw, ema):
        if report["checkpoint_step"] != 2000:
            failures.append("Wrong checkpoint in pilot evaluation")
        if report["cases"] != 434 or len(report["results"]) != 434 * 3 * 12:
            failures.append("Incomplete pilot evaluation")
        for row in report["results"]:
            if row["method"] == "residual" and row["protected_max_error"] not in (None, 0):
                failures.append("Residual changed a protected pixel")
                break
    selected = [
        row
        for row in raw["summary"]
        if row["subset"] in ("matched_real", "geometry") and row["denoise_steps"] == 5
    ]
    if len(selected) != 24:
        failures.append("Missing raw line-quality summaries")
    for row in selected:
        if row["ink_f1"] < 0.3:
            failures.append(f"Possible collapse: {row['method']}/{row['subnet']}/{row['subset']}")
    return dict(
        passed=not failures,
        failures=failures,
        criteria=(
            "Complete raw/EMA evaluation, zero residual protected-pixel error, raw F1 >= 0.30 "
            "in matched real and geometry for every subnet at five denoising steps. "
            "This is an execution/collapse gate, not a quality acceptance threshold."
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--attach-pilot", action="store_true", help="Wait for an already-running 2,000-step pilot"
    )
    args = parser.parse_args()
    root = args.root
    config, group, cases = root / "config.toml", root / "training", root / "evaluation"
    stop = root / "stop"
    if not config.is_file() or not (cases / "cases.json").is_file():
        parser.error("Prepare root/config.toml and the fixed evaluation cases first")

    def status(stage, **extra):
        record = dict(stage=stage, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra)
        (root / "workflow-status.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)

    def run(script, arguments, log):
        status(log)
        with (root / f"{log}.log").open("w") as output:
            subprocess.run(
                [sys.executable, "-u", f"scripts/{script}.py", *map(str, arguments)],
                stdout=output,
                stderr=subprocess.STDOUT,
                check=True,
            )

    def progress():
        path = group / "progress.json"
        return json.loads(path.read_text())["step"] if path.is_file() else 0

    common = ["--study", "methods", "--config", config, "--run-dir", group, "--stop-file", stop]
    try:
        if args.attach_pilot:
            status("waiting_for_pilot")
            while "Completed paired step 2000" not in (root / "training-2000.log").read_text():
                if stop.exists():
                    status("stop_requested")
                    return
                log = (root / "training-2000.log").read_text()
                if (
                    "Traceback (most recent call last)" in log
                    or time.time() - (root / "training-2000.log").stat().st_mtime > 1200
                ):
                    raise RuntimeError(
                        "Pilot failed or stopped reporting progress; inspect its log"
                    )
                time.sleep(30)
        elif progress() < 2000:
            extra = ["--resume"] if progress() else []
            run(
                "compare_nested_training",
                [*common, *extra, "--stop-after", 2000 - progress()],
                "training-2000",
            )
        if stop.exists():
            status("stop_requested")
            return
        if not (root / "noise-assignment.json").is_file():
            run(
                "diagnose_noise_assignment",
                ["--output", root / "noise-assignment.json"],
                "noise-assignment",
            )
        for step in (2000, 10000):
            if step == 10000 and progress() < step:
                gate = json.loads((root / "pilot-gate.json").read_text())
                if not gate["passed"]:
                    status("pilot_needs_review", failures=gate["failures"])
                    return
                if stop.exists():
                    status("stop_requested")
                    return
                run(
                    "compare_nested_training",
                    [*common, "--resume", "--stop-after", step - progress()],
                    "training-10000",
                )
                if stop.exists():
                    status("stop_requested")
                    return
            for mode in ("raw", "ema"):
                destination = cases / f"step-{step}-{mode}" / "metrics.json"
                if destination.is_file():
                    report = json.loads(destination.read_text())
                    if report["cases"] == 434:
                        continue
                flags = ["--raw"] if mode == "raw" else []
                run(
                    "evaluate_training_methods",
                    [
                        "--config",
                        config,
                        "--cases",
                        cases,
                        "--run-group",
                        group,
                        "--step",
                        step,
                        *flags,
                    ],
                    f"evaluation-{step}-{mode}",
                )
            if step == 2000:
                reports = [
                    json.loads((cases / f"step-2000-{mode}" / "metrics.json").read_text())
                    for mode in ("raw", "ema")
                ]
                gate = pilot_gate(*reports)
                (root / "pilot-gate.json").write_text(json.dumps(gate, indent=2) + "\n")
                status("pilot_evaluated", gate=gate)
        status("complete", optimizer_steps=progress(), evaluated_steps=[2000, 10000])
    except Exception as error:
        status("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
