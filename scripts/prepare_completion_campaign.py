"""Freeze the campaign inputs and numerical source before its first update."""

import argparse
import re
import shutil
import subprocess
from pathlib import Path

from campaign_protocol import read, sha, write

from lineartgen.config import load_config, save_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    assert not (root / "arms").exists(), "Freeze inputs before training"
    assert re.search(r"168 passed", (root / "tests.log").read_text())
    assert "2 passed" in (root / "protocol-tests.log").read_text()
    assert "1 passed" in (root / "workflow-final.log").read_text()
    preflight = read(root / "preflight.json")
    assert preflight["original_default_recursion_bit_exact"] and len(preflight["records"]) == 36
    for source, name in (
        (Path("configs/completion-campaign.toml"), "base.toml"),
        (Path("configs/completion-protocol.json"), "protocol.json"),
    ):
        target = root / name
        if target.exists():
            assert sha(source) == sha(target)
        else:
            shutil.copyfile(source, target)
    save_config(load_config(root / "base.toml"), root / "config.json")
    source = Path("runs/mask-sampling-20260918/evaluation")
    hashes = read(source.parent / "evaluation-input-sha256.json")
    cases = root / "development"
    cases.mkdir(exist_ok=True)
    for name, digest in hashes.items():
        assert sha(source / name) == digest
        if not (cases / name).exists():
            shutil.copyfile(source / name, cases / name)
        assert sha(cases / name) == digest
    assert len(read(cases / "cases.json")["cases"]) == 1562
    write(root / "development-input-sha256.json", hashes)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    archive = root / "training-source.tar.gz"
    assert not archive.exists()
    subprocess.run(["git", "archive", "--format=tar.gz", "-o", str(archive), commit], check=True)
    files = [
        *Path("src").rglob("*.py"),
        Path("scripts/compare_nested_training.py"),
        Path("scripts/evaluate_training_methods.py"),
        Path("scripts/campaign_protocol.py"),
        Path("configs/completion-protocol.json"),
        Path("configs/completion-campaign.toml"),
        Path("pyproject.toml"),
        Path("uv.lock"),
        root / "base.toml",
        root / "config.json",
        root / "protocol.json",
    ]
    write(
        root / "training-source.json",
        dict(
            commit=commit,
            archive=str(archive),
            archive_sha256=sha(archive),
            file_sha256={str(p): sha(p) for p in files},
        ),
    )
    write(
        root / "execution-source.json",
        dict(
            commit=commit,
            files={str(p): sha(p) for p in Path("scripts").glob("*.py")},
            note="Numerical source and rules are frozen separately. "
            "Later reporting and final-stage additions record source before execution.",
        ),
    )
    print(root / "training-source.json")


if __name__ == "__main__":
    main()
