"""Audit checkpoints before retention removes them, then run fixed supplementary diagnostics."""

import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

from campaign_protocol import read, sha, write
from train_lineart_loss_study import verify

from lineartgen.training.lineart_study import METHODS


def analysis_sources(root):
    for path, expected in read(root / "analysis-source.json").items():
        assert sha(Path(path)) == expected, f"Analysis source changed: {path}"


def status(root, stage, **extra):
    write(
        root / "analysis-status.json",
        dict(
            stage=stage,
            pid=os.getpid(),
            utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **extra,
        ),
    )


def assert_running(root):
    if (root / "stop").exists():
        raise RuntimeError("Study stop file present; supplementary watcher stopped")
    state = read(root / "workflow-status.json")
    if state["stage"] in ("failed", "stopped_by_stop_file"):
        raise RuntimeError(f"Training controller requires review: {state}")
    pid = read(root / "launch.json")["pid"]
    os.kill(pid, 0)
    command = Path(f"/proc/{pid}/cmdline").read_bytes()
    if b"run_lineart_loss_study.py" not in command:
        raise RuntimeError(f"Controller PID {pid} has been reused")


def wait_for(root, predicate, stage):
    status(root, stage)
    while not predicate():
        assert_running(root)
        time.sleep(30)


def run(root):
    protocol = verify(root)
    analysis_sources(root)

    def command(script, arguments, *, gpu=False):
        if (root / "stop").exists():
            raise RuntimeError("Study stop file present")
        analysis_sources(root)
        label = script + "-" + "-".join(map(str, arguments))
        status(root, label)
        environment = os.environ | dict(
            JAX_PLATFORMS="rocm,cpu" if gpu else "cpu",
            OPENBLAS_NUM_THREADS="1",
            XLA_PYTHON_CLIENT_PREALLOCATE="false",
            JAX_COMPILATION_CACHE_DIR=".cache/lineart-loss-compilation-20261006",
        )
        logs = root / "analysis-logs"
        logs.mkdir(exist_ok=True)
        with (logs / f"{label}.log").open("a") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    f"scripts/{script}.py",
                    "--root",
                    str(root),
                    *map(str, arguments),
                ],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    for step in protocol["milestones"]:
        audit_path = root / "checkpoint-audits" / f"step-{step}.json"
        if not audit_path.exists():
            wait_for(
                root,
                lambda step=step: all(
                    (root / f"snapshots/step-{step}" / name / mode / "weights.npz").is_file()
                    for name in METHODS
                    for mode in ("raw", "ema")
                ),
                f"waiting_for_snapshots_{step}",
            )
            command("audit_lineart_loss_checkpoint", ["--step", step, "--snapshots"])
        audit = read(audit_path)
        assert audit["step"] == step and audit["snapshots_checked"]
        assert audit["analysis_source_sha256"] == sha(
            Path("scripts/audit_lineart_loss_checkpoint.py")
        )
        wait_for(
            root,
            lambda step=step: all(
                (root / f"evaluation/step-{step}-{mode}/metrics.json").is_file()
                for mode in ("raw", "ema")
            ),
            f"waiting_for_evaluation_{step}",
        )
        command("summarize_lineart_loss_study", ["--through", step])
        for mode in ("ema", "raw"):
            command("evaluate_lineart_blank_study", ["--step", step, "--mode", mode], gpu=True)
            command("render_lineart_loss_supplement", ["--step", step, "--mode", mode], gpu=True)
    status(root, "numerical_diagnostics_complete_visual_review_pending")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    with (root / "analysis-watcher.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(root)
        except Exception as error:
            status(root, "failed", error=str(error))
            raise


if __name__ == "__main__":
    main()
