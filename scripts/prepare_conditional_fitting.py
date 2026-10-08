"""Prepare the matched rough/bridge fitting probe and its historical replay."""

import argparse
import shutil
from pathlib import Path

import jax
import numpy as np
import orbax.checkpoint as ocp
from campaign_protocol import read, sha, write
from diagnose_learnability import arms_from_root, verify
from flax import nnx

from lineartgen.config import ExperimentConfig, load_config, save_config
from lineartgen.evaluation_sampling import inference_with_source, runtime_inputs
from lineartgen.models.serialization import load_inference
from lineartgen.training import lineart_state


def prepare(root, parent):
    verify(parent)
    if root.exists():
        raise FileExistsError(root)
    old = read(parent / "plan.json")
    assert old["updates"] == 1200 and old["seed"] == 2026100801
    base = load_config(parent / "configs/warm_detail16.json")
    assert base.model.detail_channels == 16 and base.model.source_conditioning
    assert not base.model.progress_conditioning and base.diffusion.training_path == "rough"
    assert base.loss.pixel_loss == "clipped_mse" and base.train.steps == 1200
    for destination, paths in (
        (root, {"progress_rough": "rough", "progress_bridge": "bridge"}),
        (root / "baseline-replay", {"warm_detail16": "rough"}),
    ):
        (destination / "configs").mkdir(parents=True)
        plan = dict(old)
        for split in ("train", "probe"):
            for suffix in ("json", "npz"):
                shutil.copyfile(parent / f"{split}.{suffix}", destination / f"{split}.{suffix}")
            assert sha(destination / f"{split}.npz") == old["data"][split]["sha256"]
        plan.update(
            arms=list(paths),
            warm_start_methods=list(paths),
            counterparts={n: "warm_detail16" for n in paths},
            detail_channels={n: 16 for n in paths},
            paths=paths,
            parent=str(parent),
            parent_source_sha256=sha(parent / "source.json"),
            plan_document_sha256=sha(Path("docs/conditional-path-fitting-plan-20261008.md")),
            data={
                s: dict(old["data"][s], path=str(destination / f"{s}.npz"))
                for s in ("train", "probe")
            },
            primary="raw",
            primary_steps=5,
            evaluation_steps=[1, 2, 5],
            advance_rule=(
                "Fixed paired five-step and original one-step local gates in the plan "
                "document; native review required; no adoption"
            ),
            campaign_confirmation_generated=False,
            reference=dict(
                method="warm_detail16",
                step=1200,
                checkpoints={
                    mode: dict(
                        path=str(parent / f"snapshots/step-1200/warm_detail16/{mode}"),
                        files={
                            n: sha(parent / f"snapshots/step-1200/warm_detail16/{mode}/{n}")
                            for n in ("weights.npz", "config.json", "metadata.json")
                        },
                    )
                    for mode in ("raw", "ema")
                },
            ),
        )
        write(destination / "plan.json", plan)
        for name, path in paths.items():
            values = base.model_dump()
            if destination == root:
                values["model"]["progress_conditioning"] = True
                values["diffusion"].update(
                    training_path=path, progress_minimum=0.05, progress_initial_probability=0.5
                )
            updated = ExperimentConfig.model_validate(values)
            save_config(updated, destination / f"configs/{name}.json")
    left, right = (read(root / f"configs/{n}.json") for n in ("progress_rough", "progress_bridge"))
    left["diffusion"]["training_path"] = "bridge"
    assert left == right
    print(dict(root=str(root), data_exact=True, only_paired_difference="training_path"), flush=True)


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
        return {"warm_detail16": arms["warm_detail16"]}

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
    print(dict(historical_width16_replay_exact=True), flush=True)


def audit_initial(root):
    verify(root)
    original, _ = load_inference(read(root / "plan.json")["initial"])
    arms = arms_from_root(root)
    inference = nnx.jit(
        lambda m, x, t, r, n: inference_with_source(m, x, t, r, steps=n), static_argnums=4
    )
    source = np.ones((1, 512, 512, 1), np.float32)
    source[:, 250:254] = 0.1
    source[:, :, 190:194] = -0.4
    rows = []
    for levels in range(1, 7):
        strength = np.full_like(source, 0.1 if levels % 2 else 0.5)
        strength[:, :103] = 0
        strength[:, :, :171] = 0
        inputs = runtime_inputs(source, strength, 42, levels)
        for steps in (1, 2, 5):
            baseline = np.asarray(inference(original, *inputs, source, steps))
            for name, arm in arms.items():
                actual = np.asarray(inference(arm["model"], *inputs, source, steps))
                np.testing.assert_array_equal(actual, baseline)
                rows.append(
                    dict(
                        method=name,
                        levels=levels,
                        steps=steps,
                        strength=0.1 if levels % 2 else 0.5,
                        exact=True,
                    )
                )
    write(
        root / "initial-forward-audit.json",
        dict(rows=rows, all_exact=True, source_sha256=sha(root / "source.json")),
    )
    print(dict(initial_forward_comparisons=len(rows), all_exact=True), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "audit-replay", "audit-initial"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--parent", type=Path)
    a = p.parse_args()
    if a.action == "prepare":
        prepare(a.root.resolve(), a.parent.resolve())
    elif a.action == "audit-replay":
        audit_replay(a.root.resolve())
    else:
        audit_initial(a.root.resolve())
