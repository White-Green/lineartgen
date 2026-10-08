"""Compare legacy-strokes guided baseline states against completed six-depth updates."""

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
    initial, _ = load_inference(protocol["initial"])

    def restore(directory):
        names = study_methods(read(directory / "protocol.json"))
        arms = {n: create_arm(load_config(directory / f"configs/{n}.json"), initial) for n in names}
        assert all(
            a["config"].model.source_conditioning and a["config"].sketch.mode == "strokes"
            for a in arms.values()
        )
        with ocp.CheckpointManager(
            (directory / "preflight/checkpoints").resolve(),
            options=ocp.CheckpointManagerOptions(read_only=True),
        ) as saved:
            assert lineart_state.restore(saved, arms, 6) == 6
        return arms

    replay = restore(root / "baseline-replay")
    records = []
    for declaration in protocol["replay_parents"]:
        parent = Path(declaration["root"])
        all_arms = restore(parent)
        arms = {name: all_arms[name] for name in declaration["methods"]}
        jax.tree.map(
            np.testing.assert_array_equal,
            lineart_state.payload(arms, 6),
            lineart_state.payload({n: replay[n] for n in arms}, 6),
        )
        records.append(declaration | dict(preflight_sha256=sha(parent / "preflight/results.json")))
    write(
        root / "baseline-replay-audit.json",
        dict(
            full_model_optimizer_ema_exact=True,
            step=6,
            methods=list(replay),
            parents=records,
            replay_preflight_sha256=sha(root / "baseline-replay/preflight/results.json"),
            script_sha256=sha(Path(__file__)),
        ),
    )
    print("Both guided baseline model/optimizer/EMA states reproduce exactly after six GPU updates")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    audit(p.parse_args().root.resolve())
