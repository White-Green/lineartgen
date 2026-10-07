"""Read-only restore audit of actual study counters, finite states and inference snapshots."""

import argparse
from pathlib import Path

import jax
import numpy as np
import orbax.checkpoint as ocp
from campaign_protocol import sha, write
from flax import nnx
from train_lineart_loss_study import verify

from lineartgen.config import load_config
from lineartgen.models.serialization import load_inference
from lineartgen.training import lineart_state
from lineartgen.training.lineart_study import create_arm, study_methods


def audit(root, step, snapshots):
    protocol = verify(root)
    initial, _ = load_inference(protocol["initial"])
    arms = {
        name: create_arm(
            load_config(root / "configs" / f"{name}.json"),
            initial,
            adversarial=name == "patch_prior",
            patch_size=protocol["patch_size"],
        )
        for name in study_methods(protocol)
    }
    with ocp.CheckpointManager(
        (root / "checkpoints").resolve(),
        options=ocp.CheckpointManagerOptions(read_only=True),
    ) as saved:
        restored_step = lineart_state.restore(saved, arms, step)
        assert restored_step == step
    payload = lineart_state.payload(arms, step)
    records = []
    for name, arm in arms.items():
        state = payload["arms"][name]
        assert int(state["step"]) == step
        assert int(state["optimizer"]["step"]) == step
        assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(state))
        parameters = sum(np.size(x) for x in jax.tree.leaves(nnx.state(arm["model"], nnx.Param)))
        record = dict(
            method=name, generator_step=step, parameters=parameters, all_states_finite=True
        )
        if arm["critic"] is not None:
            assert int(state["critic_optimizer"]["step"]) == step
            record["critic_step"] = step
            record["critic_parameters"] = sum(
                np.size(x) for x in jax.tree.leaves(nnx.state(arm["critic"], nnx.Param))
            )
        if snapshots:
            record["snapshots"] = {}
            for mode in ("raw", "ema"):
                path = root / "snapshots" / f"step-{step}" / name / mode
                exported, _ = load_inference(path)
                expected = nnx.state(arm["model"], nnx.Param) if mode == "raw" else arm["ema"]
                jax.tree.map(
                    np.testing.assert_array_equal, nnx.state(exported, nnx.Param), expected
                )
                record["snapshots"][mode] = dict(
                    checkpoint_exact=True,
                    weights_sha256=sha(path / "weights.npz"),
                )
        records.append(record)
    result = dict(
        step=step,
        records=records,
        read_only_restore=True,
        snapshots_checked=snapshots,
        devices=[str(x) for x in jax.devices()],
        analysis_source_sha256=sha(Path(__file__)),
    )
    write(root / "checkpoint-audits" / f"step-{step}.json", result)
    print(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--snapshots", action="store_true")
    args = parser.parse_args()
    audit(args.root.resolve(), args.step, args.snapshots)


if __name__ == "__main__":
    main()
