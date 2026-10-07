"""Verify disabled paper loss leaves completed six-depth STE updates exactly unchanged."""

import argparse
from pathlib import Path

import jax
import numpy as np
import orbax.checkpoint as ocp
from campaign_protocol import read, sha, write

from lineartgen.config import load_config
from lineartgen.models.serialization import load_inference
from lineartgen.training import lineart_state
from lineartgen.training.lineart_study import create_arm, study_methods


def audit(root, parent):
    initial, _ = load_inference(read(root / "protocol.json")["initial"])
    states = []
    for directory in (parent, root / "baseline-replay"):
        names = study_methods(read(directory / "protocol.json"))
        assert names == ("ste_control", "ste_ink_guard")
        arms = {n: create_arm(load_config(directory / f"configs/{n}.json"), initial) for n in names}
        assert all(a["config"].loss.paper_peak_weight == 0 for a in arms.values())
        with ocp.CheckpointManager(
            (directory / "preflight/checkpoints").resolve(),
            options=ocp.CheckpointManagerOptions(read_only=True),
        ) as saved:
            assert lineart_state.restore(saved, arms, 6) == 6
        states.append(lineart_state.payload(arms, 6))
    jax.tree.map(np.testing.assert_array_equal, *states)
    write(
        root / "baseline-replay-audit.json",
        dict(
            full_model_optimizer_ema_exact=True,
            step=6,
            methods=list(names),
            parent_preflight_sha256=sha(parent / "preflight/results.json"),
            replay_preflight_sha256=sha(root / "baseline-replay/preflight/results.json"),
            script_sha256=sha(Path(__file__)),
            limitation="Six actual GPU updates, one per depth, not an independent fine-tuning seed",
        ),
    )
    print("Both STE baselines reproduce all six updates and full states exactly")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    args = parser.parse_args()
    audit(args.root.resolve(), args.parent.resolve())
