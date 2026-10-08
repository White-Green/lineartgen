"""Seal committed numerical code, settings and validation records before training."""

import argparse
import importlib.metadata
import shutil
import subprocess
from pathlib import Path

from campaign_protocol import read, sha, write

from lineartgen.training.lineart_study import study_methods


def freeze(root, tests_log):
    if (root / "source.json").exists():
        raise FileExistsError("Refusing to replace a frozen source manifest")
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    log = tests_log.read_text()
    assert "passed" in log and "failed" not in log and "ERROR" not in log
    validation = read(root / "preflight/results.json")
    methods = study_methods(read(root / "protocol.json"))
    assert validation["checkpoint_all_states_exact"] and validation["common_initial_weights_exact"]
    assert len(validation["updates"]) == len(methods) * 6
    assert all("cpu" not in d.lower() for d in validation["devices"])
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(
        ["git", "archive", "--format=tar.gz", "-o", str(root / "source.tar.gz"), commit], check=True
    )
    tracked = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    files = [Path(p) for p in tracked if p.endswith((".py", ".toml")) or p == "uv.lock"]
    settings = [
        root / n
        for n in [
            "protocol.json",
            "config.json",
            "dataset.json",
            "evaluation.json",
            "calibration.json",
            "acceptance.json",
            "supplemental-panels.json",
            "confirmation-plan.json",
        ]
    ]
    settings.extend(sorted((root / "configs").glob("*.json")))
    settings.extend(
        root / name
        for name in (
            "plan.json",
            "baseline-replay-audit.json",
            "baseline-replay/preflight/results.json",
            "initial-guide-audit.json",
            "mixture-diagnostic.json",
            "supplemental-validation.json",
        )
        if (root / name).is_file()
    )
    shutil.copyfile(tests_log, root / "tests.log")
    write(
        root / "source.json",
        dict(
            commit=commit,
            files={str(p): sha(p) for p in files},
            settings={str(p.relative_to(root)): sha(p) for p in settings},
            archive_sha256=sha(root / "source.tar.gz"),
            versions={
                name: importlib.metadata.version(name)
                for name in ["jax", "jaxlib", "flax", "optax", "orbax-checkpoint", "numpy"]
            },
            tests=dict(
                log_sha256=sha(root / "tests.log"),
                preflight_sha256=sha(root / "preflight/results.json"),
            ),
        ),
    )
    analysis = [
        Path("scripts") / (name + ".py")
        for name in [
            "summarize_lineart_loss_study",
            "evaluate_lineart_blank_study",
            "render_lineart_loss_supplement",
            "audit_lineart_loss_checkpoint",
            "watch_lineart_loss_analysis",
            "assess_ink_structure_study",
        ]
    ]
    analysis.append(root / "supplemental-panels.json")
    write(root / "analysis-source.json", {str(p): sha(p) for p in analysis})
    print(dict(commit=commit, source_files=len(files), settings=len(settings), methods=methods))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tests-log", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.root.resolve(), args.tests_log)
