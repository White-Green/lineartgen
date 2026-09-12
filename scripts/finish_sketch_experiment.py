"""Package the pilot and final evidence after the existing controller produces it.

This observer never starts or restarts training. It exits on controller failure or
a requested stop, and requires the original controller to remain alive while waiting.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--controller-pid", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/sketch-20260911.json"))
    parser.add_argument(
        "--summary-script",
        choices=["summarize_sketch_experiment", "summarize_sketch_rates"],
        default="summarize_sketch_experiment",
    )
    args = parser.parse_args()
    root = args.root.resolve()

    def status(stage, **extra):
        record = dict(stage=stage, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra)
        temporary = root / "finish-status.partial.json"
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(root / "finish-status.json")
        print(json.dumps(record), flush=True)

    def package(step, destination):
        status(f"packaging_{step}")
        env = dict(os.environ, JAX_PLATFORMS="cpu", OPENBLAS_NUM_THREADS="1")
        with (root / f"summary-{step}.log").open("a") as output:
            subprocess.run(
                [
                    sys.executable,
                    f"scripts/{args.summary_script}.py",
                    "--root",
                    str(root),
                    "--through",
                    str(step),
                    "--output",
                    str(destination),
                ],
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=True,
            )

    pilot = args.output.with_name(args.output.stem + "-pilot.json")
    try:
        status("waiting_for_pilot")
        while True:
            state_path = root / "workflow-status.json"
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            if state.get("stage") in ("failed", "stop_requested", "pilot_needs_review"):
                status("controller_stopped", controller=state)
                return
            if (root / "pilot-gate.json").is_file() and not pilot.exists():
                package(2000, pilot)
                status("waiting_for_final")
            if state.get("stage") == "complete":
                package(10000, args.output)
                status("artifacts_verified", output=str(args.output))
                return
            # This PID is the observed Python controller, never an inferred replacement.
            if not Path(f"/proc/{args.controller_pid}").exists():
                # Re-read once to cover exit immediately after writing the terminal state.
                final = json.loads(state_path.read_text())
                if final.get("stage") == "complete":
                    continue
                raise RuntimeError(f"Controller exited without completion: {final}")
            time.sleep(30)
    except Exception as error:
        status("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
