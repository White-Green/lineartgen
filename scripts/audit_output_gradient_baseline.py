"""Compare replayed six-depth hard-clip model/optimizer/EMA with the completed baseline."""

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


def audit(root):
    protocol = read(root / "protocol.json")
    parent = Path(protocol["external_baselines"]["2000"]["root"])
    initial, _ = load_inference(protocol["initial"])
    states = []
    for directory in (parent, root / "hard-control-replay"):
        arms = {
            name: create_arm(load_config(directory / f"configs/{name}.json"), initial)
            for name in study_methods(read(directory / "protocol.json"))
        }
        assert all(a["config"].model.output_clip_gradient == "hard" for a in arms.values())
        checkpoint = directory / "preflight/checkpoints"
        with ocp.CheckpointManager(
            checkpoint.resolve(), options=ocp.CheckpointManagerOptions(read_only=True)
        ) as saved:
            assert lineart_state.restore(saved, arms, 6) == 6
        states.append(lineart_state.payload({n: arms[n] for n in ("control", "ink_guard")}, 6))
    jax.tree.map(np.testing.assert_array_equal, *states)
    write(
        root / "hard-baseline-replay-audit.json",
        dict(
            full_model_optimizer_ema_exact=True,
            step=6,
            methods=["control", "ink_guard"],
            parent_preflight_sha256=sha(parent / "preflight/results.json"),
            replay_preflight_sha256=sha(root / "hard-control-replay/preflight/results.json"),
            script_sha256=sha(Path(__file__)),
            limitation="Six GPU updates, one per recursion depth; not a second training seed",
        ),
    )
    print("Both hard baselines reproduce all six updates and full states exactly")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    audit(args.root.resolve())
