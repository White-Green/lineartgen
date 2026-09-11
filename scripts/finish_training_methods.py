"""Wait for the four-method workflow, then verify and summarize its final artifacts."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def read_if_ready(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        # The training controller writes its small status files in place.
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    if not (root / "config.toml").is_file():
        parser.error("Prepare the experiment root first")

    def status(stage, **extra):
        record = dict(stage=stage, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra)
        temporary = root / "finish-status.partial.json"
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(root / "finish-status.json")
        print(json.dumps(record), flush=True)

    def run(script, arguments, log, *, cpu=False):
        status(log)
        environment = os.environ.copy()
        if cpu:
            environment["JAX_PLATFORMS"] = "cpu"
        with (root / f"{log}.log").open("w") as output:
            subprocess.run(
                [sys.executable, "-u", f"scripts/{script}.py", *map(str, arguments)],
                stdout=output,
                stderr=subprocess.STDOUT,
                env=environment,
                check=True,
            )

    try:
        pilot_recorded = False
        status("waiting_for_evaluations")
        while True:
            if (root / "stop").exists():
                status("stop_requested")
                return
            gate = read_if_ready(root / "pilot-gate.json")
            if gate is not None and not pilot_recorded:
                pilot_output = args.output.with_name(args.output.stem + "-pilot.json")
                run(
                    "summarize_training_methods",
                    ["--root", root, "--output", pilot_output, "--through", 2000],
                    "summary-2000",
                    cpu=True,
                )
                pilot_recorded = True
                status("waiting_for_final_evaluation", pilot_summary=str(pilot_output))
            workflow = read_if_ready(root / "workflow-status.json")
            if workflow is not None:
                if workflow["stage"] in ("failed", "pilot_needs_review", "stop_requested"):
                    status("workflow_needs_attention", workflow=workflow)
                    return
                if workflow["stage"] == "complete":
                    assert workflow["optimizer_steps"] == 10000
                    assert workflow["evaluated_steps"] == [2000, 10000]
                    assert pilot_recorded
                    break
            time.sleep(30)
        for script, flags, log, cpu in (
            ("verify_training_methods", ["--check-checkpoints"], "verify-10000", False),
            ("verify_training_methods", ["--exports-only"], "export-10000", True),
            ("summarize_training_methods", ["--output", args.output], "summary-10000", True),
        ):
            if (root / "stop").exists():
                status("stop_requested")
                return
            run(script, ["--root", root, *flags], log, cpu=cpu)
        status("artifacts_verified", summary=str(args.output))
    except Exception as error:
        status("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
