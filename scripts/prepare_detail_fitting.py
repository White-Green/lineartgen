"""Copy the fixed fitting task, adding only the full-resolution detail branch."""

import argparse
import shutil
from pathlib import Path

import jax
import numpy as np
import orbax.checkpoint as ocp
from campaign_protocol import read, sha, write
from diagnose_learnability import arms_from_root, verify

from lineartgen.config import load_config, save_config
from lineartgen.training import lineart_state


def prepare(root, parent):
    verify(parent)
    if root.exists():
        raise FileExistsError(root)
    old = read(parent / "plan.json")
    assert old["updates"] == 1200 and old["seed"] == 2026100801
    for destination, methods, widths in (
        (
            root,
            {"warm_detail8": "warm_hard", "warm_detail16": "warm_hard"},
            {"warm_detail8": 8, "warm_detail16": 16},
        ),
        (
            root / "baseline-replay",
            {"warm_hard": "warm_hard"},
            {"warm_hard": 0},
        ),
    ):
        (destination / "configs").mkdir(parents=True)
        plan = dict(old)
        for split in ("train", "probe"):
            for suffix in ("json", "npz"):
                shutil.copyfile(parent / f"{split}.{suffix}", destination / f"{split}.{suffix}")
            assert sha(destination / f"{split}.npz") == old["data"][split]["sha256"]
        plan["data"] = {
            s: dict(old["data"][s], path=str(destination / f"{s}.npz")) for s in ("train", "probe")
        }
        plan.update(
            arms=list(methods),
            counterparts=methods,
            detail_channels=widths,
            parent=str(parent),
            parent_source_sha256=sha(parent / "source.json"),
            plan_document_sha256=sha(Path("docs/detail-head-fitting-plan-20261008.md")),
            advance_rule="Raw final blur F1 gain >=.03 in both splits; clean F1 drop <=.01; "
            "no family/split halo increase >.01; native visual support. No adoption.",
        )
        write(destination / "plan.json", plan)
        for name, counterpart in methods.items():
            base = load_config(parent / f"configs/{counterpart}.json")
            assert base.model.residual_scale_floor == base.model.detail_channels == 0
            updated = base.model_copy(
                update={"model": base.model.model_copy(update={"detail_channels": widths[name]})}
            )
            save_config(updated, destination / f"configs/{name}.json")
    print(dict(root=str(root), copied_exact=True), flush=True)


def audit_replay(root):
    parent = Path(read(root / "plan.json")["parent"])
    verify(parent)

    def restore(directory):
        arms = arms_from_root(directory)
        with ocp.CheckpointManager(
            (directory / "preflight/checkpoints").resolve(),
            options=ocp.CheckpointManagerOptions(read_only=True),
        ) as saved:
            assert lineart_state.restore(saved, arms, 6) == 6
        return {name: arms[name] for name in ("warm_hard",)}

    before, after = restore(parent), restore(root / "baseline-replay")
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(before, 6),
        lineart_state.payload(after, 6),
    )
    write(
        root / "baseline-replay-audit.json",
        dict(
            step=6,
            full_state_exact=True,
            parent=str(parent),
            parent_source_sha256=sha(parent / "source.json"),
            replay_source_sha256=sha(root / "baseline-replay/source.json"),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "audit-replay"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--parent", type=Path)
    a = p.parse_args()
    if a.action == "prepare":
        prepare(a.root.resolve(), a.parent.resolve())
    else:
        audit_replay(a.root.resolve())
