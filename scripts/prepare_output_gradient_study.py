"""Freeze a paired two-arm surrogate-gradient screen using completed hard-clip baselines."""

import argparse
import shutil
from pathlib import Path

from campaign_protocol import read, sha, write

from lineartgen.config import load_config, save_config


def prepare(root, screen, continuation):
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(root)
    root.mkdir(parents=True)
    parent_worktree = screen.parent.parent
    for name in (
        "dataset.json",
        "evaluation.json",
        "calibration.json",
        "acceptance.json",
        "supplemental-panels.json",
        "confirmation-plan.json",
    ):
        shutil.copyfile(screen / name, root / name)

    def relocated(config):
        changes = {}
        for name in ("root", "prepared_root"):
            value = getattr(config.data, name)
            if value is not None:
                changes[name] = (parent_worktree / value).resolve()
        return config.model_copy(update={"data": config.data.model_copy(update=changes)})

    base = relocated(load_config(screen / "config.json"))
    save_config(base, root / "config.json")
    mappings = {"ste_control": "control", "ste_ink_guard": "ink_guard"}
    (root / "configs").mkdir()
    for method, counterpart in mappings.items():
        config = relocated(load_config(screen / f"configs/{counterpart}.json"))
        assert config.model.output_clip_gradient == "hard"
        config = config.model_copy(
            update={
                "model": config.model.model_copy(
                    update={"output_clip_gradient": "straight_through"}
                )
            }
        )
        save_config(config, root / f"configs/{method}.json")
    external = {}
    for step, parent in [(2000, screen), (10000, continuation)]:
        assert read(parent / "checkpoint.json")["step"] == step
        external[str(step)] = dict(
            root=str(parent),
            methods=["control", "ink_guard"],
            source_sha256=sha(parent / "source.json"),
        )
    protocol = read(screen / "protocol.json") | dict(
        methods=list(mappings),
        milestones=[2000],
        hard_counterparts=mappings,
        external_baselines=external,
        optimizer="Reset optimizer/EMA from same 100k EMA; unchanged 10k cosine clock",
        intervention="Only output Clip backward becomes identity; forward remains exact Clip",
        calibration="No loss recalibration: reuse fixed control and ink_guard coefficients",
        next_phase="Compare paired external baselines, inspect 2k. Continue fixed-gate passing "
        "candidates to 10k with all states retained, then independent confirmations.",
    )
    write(root / "protocol.json", protocol)
    write(
        root / "diagnostic-plan.json",
        dict(
            training_saturation_seed=20262006,
            training_saturation_batches=12,
            training_depths=[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
            reason="Hard clip blocks direct gradients in observed missing-line pixels. "
            "Measure training occurrence; test backward-only change, not more loss coefficients.",
            clipping_scope="All recursion levels; identity surrogate, explicitly biased",
            safeguards="Same loss values and inference at equal weights, frozen inputs/parameters, "
            "six-depth GPU updates, raw/EMA/checkpoint equality, finite gradients and ONNX parity",
            acceptance="Unchanged inherited screen/final gates; no exceptions for this hypothesis",
        ),
    )
    print(dict(root=str(root), methods=list(mappings), external=external))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--continuation", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root.resolve(), args.screen.resolve(), args.continuation.resolve())
