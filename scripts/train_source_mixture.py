"""Paired blur/mixed synthetic rough training on held-out-source-separated clean art."""

import argparse
import fcntl
import hashlib
import json
import shutil
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from campaign_protocol import read, sha, write
from diagnose_learnability import arms_from_root, freeze, snapshot, update, verify
from flax import nnx
from train_connection_queries import evaluate as evaluate_primary
from train_lineart_loss_study import restore_journal
from train_procedural_structure import prepare_batch

from lineartgen.config import load_config, save_config
from lineartgen.data.geometry import mix_geometry
from lineartgen.data.pipeline import batches, dataset_description, discover, split_experiment
from lineartgen.diffusion import recursive_forward_with_preclip
from lineartgen.models.serialization import load_inference
from lineartgen.training import checkpoints, lineart_state
from lineartgen.training.losses import image_loss

WORKTREES = Path("/home/whitegreen/projects/lineartgen/.worktrees")
SEED = 2026100961
STEPS = 4000
METHODS = ("pixel", "mixed")


def prepare(root):
    assert not root.exists()
    parent = WORKTREES / "jax-source-signal-20261009/runs/source-signal-20261009"
    verify(parent)
    initial = parent / "snapshots/step-4000/signal/raw"
    base = load_config(initial / "config.json")
    data_config = load_config(
        WORKTREES / "jax-detail-study-20261008/runs/detail-mixed-20261008-screen/config.json"
    )
    base = base.model_copy(
        update={
            "model": base.model.model_copy(update={"sampling_mode": "one_pass"}),
            "data": data_config.data,
            "geometry": data_config.geometry,
            "train": base.train.model_copy(
                update={
                    "steps": STEPS,
                    "seed": SEED,
                    "batch_size": 4,
                    "accumulate_steps": 1,
                    "learning_rate": 5e-5,
                    "minimum_learning_rate": 1e-5,
                    "warmup_steps": 100,
                    "checkpoint_every": 100,
                    "log_every": 25,
                }
            ),
        }
    )
    assert base.train.peak_learning_rate == 1e-4
    assert base.model.feature_input == "source" and base.model.residual_base == "source"
    assert base.diffusion.scale_loss_multiplier == 0 and base.diffusion.pattern_weights[3] == 0
    assert base.loss.opacity_deficit_weight == 0.3 and base.loss.opacity_deficit_preclip
    (root / "configs").mkdir(parents=True)
    for name, mode in (("pixel", "blur"), ("mixed", "mixed")):
        config = base.model_copy(
            update={"sketch": base.sketch.model_copy(update={"mode": mode, "probability": 0.8})}
        )
        save_config(config, root / f"configs/{name}.json")
    for name in ("train.json", "probe.json", "inputs.npz", "queries.json"):
        shutil.copyfile(parent / name, root / name)
    sources = discover(base.data)
    training, validation, final = split_experiment(sources, base.data, SEED)
    description = dataset_description(sources, training, validation, final)
    expected = read(
        WORKTREES / "jax-detail-study-20261008/runs/detail-mixed-20261008-screen/dataset.json"
    )
    assert description == expected
    write(root / "dataset.json", description)
    reference = read(WORKTREES / "jax-rewrite/runs/lineart-loss-20261006/evaluation.json")
    manifest = read(Path(reference["root"]) / "cases.json")
    real_ids = {r["source"] for r in manifest["cases"] if r["kind"] == "real"}
    assert not real_ids & set(description["train"])
    assert real_ids <= set(description["validation"])
    write(root / "reference.json", reference)
    panels = read(
        WORKTREES / "jax-source-feature-broad-20261009/runs/source-feature-broad-20261009/plan.json"
    )
    groups = dict(panels["panel_groups"], partial=[717, 727])
    crops = dict(panels["crops"], **{"717": [160, 96, 320, 256], "727": [160, 96, 320, 256]})
    old = read(parent / "plan.json")
    write(
        root / "plan.json",
        dict(
            arms=list(METHODS),
            warm_start_methods=list(METHODS),
            comparison_methods=list(METHODS),
            initial=str(initial),
            initial_files={p.name: sha(p) for p in initial.iterdir() if p.is_file()},
            data=old["data"],
            parent=str(parent),
            seed=SEED,
            steps=STEPS,
            milestones=[0, 1000, STEPS],
            initial_outputs_equal=True,
            inputs_sha256=sha(root / "inputs.npz"),
            queries_sha256=sha(root / "queries.json"),
            dataset_sha256=sha(root / "dataset.json"),
            reference_sha256=sha(root / "reference.json"),
            panel_groups=groups,
            crops=crops,
            base_cases=1562,
            blank_cases=4,
            plan_document_sha256=sha(Path("docs/source-mixture-training-plan-20261009.md")),
            diagnostic_only=True,
            adoption=False,
            sealed_confirmation_generated=False,
            evaluation_source_counts={"train": 0, "validation": len(real_ids), "final_test": 0},
        ),
    )


def check(root):
    verify(root)
    plan = read(root / "plan.json")
    for name, key in [
        ("dataset.json", "dataset_sha256"),
        ("reference.json", "reference_sha256"),
        ("inputs.npz", "inputs_sha256"),
        ("queries.json", "queries_sha256"),
    ]:
        assert sha(root / name) == plan[key]
    assert sha(root / "source.tar.gz") == read(root / "source.json")["archive_sha256"]
    return plan


def make_arms(root):
    arms = arms_from_root(root)
    for arm in arms.values():
        config = arm["config"]

        def objective(model, prepared, config=config):
            predictions, physical = recursive_forward_with_preclip(
                model, prepared.noisy, prepared.noise_levels, guides=prepared.guides
            )
            return image_loss(
                predictions[0],
                prepared.targets[0],
                prepared.noise_levels[0],
                config.loss,
                config.diffusion.min_noise,
                preclip=physical[0],
            )

        def make_gradient(objective):
            @nnx.jit
            def gradient(model, prepared):
                (_, metrics), grads = nnx.value_and_grad(objective, has_aux=True)(model, prepared)
                return metrics, grads

            return gradient

        arm["gradient"] = make_gradient(objective)
    return arms


def restore_audit(root, saved, arms, step):
    restored = make_arms(root)
    assert lineart_state.restore(saved, restored, step) == step
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, step),
        lineart_state.payload(restored, step),
    )
    write(root / f"training/restore-audit-{step}.json", dict(all_state_exact=True, step=step))


def evaluate(root, arms, step):
    evaluate_primary(root, arms, step)


def train(root, end):
    check(root)
    assert end in (6, 1000, STEPS) and jax.devices()[0].platform != "cpu"
    arms = make_arms(root)
    config = arms["pixel"]["config"]
    sources = discover(config.data)
    training, validation, final = split_experiment(sources, config.data, SEED)
    assert dataset_description(sources, training, validation, final) == read(root / "dataset.json")
    preparers = {name: prepare_batch(arm["config"]) for name, arm in arms.items()}
    destination = root / "training"
    destination.mkdir(exist_ok=True)
    with checkpoints.manager(destination / "checkpoints", keep=3) as saved:
        start = lineart_state.restore(saved, arms) if saved.latest_step() is not None else 0
        restore_journal(destination, start)
        if start == 0:
            snapshot(root, arms, 0)
            evaluate(root, arms, 0)
            jax.tree.map(
                np.testing.assert_array_equal,
                lineart_state.payload({"same": arms["pixel"]}, 0),
                lineart_state.payload({"same": arms["mixed"]}, 0),
            )
        elif end > 6:
            assert read(root / "training/restore-audit-6.json")["all_state_exact"]
        stream = batches(
            training, config.data, batch_size=4, count=end - start, seed=SEED, start=start
        )
        key = jax.random.PRNGKey(SEED)
        counts = {d: 0 for d in range(1, 7)}
        try:
            for step, batch in enumerate(stream, start=start):
                begin = time.perf_counter()
                clean, records = mix_geometry(
                    batch.images,
                    config.geometry,
                    seed=SEED,
                    step=step,
                    image_scale=config.data.image_scale,
                )
                depth = config.diffusion.levels_for_scale(
                    512, config.model.minimum_size, batch.scale_level
                )
                counts[depth] += 1
                prepared = {
                    n: f(jnp.asarray(clean), jax.random.fold_in(key, step), depth)
                    for n, f in preparers.items()
                }
                if step < 6:
                    a, b = prepared["pixel"], prepared["mixed"]
                    jax.tree.map(np.testing.assert_array_equal, a.noise_levels, b.noise_levels)
                    active = np.asarray(a.noise_levels[0]) > 0
                    np.testing.assert_array_equal(
                        np.asarray(a.targets[0])[active], np.asarray(b.targets[0])[active]
                    )
                    for p in (a, b):
                        np.testing.assert_array_equal(
                            np.asarray(p.targets[0])[active], clean[active]
                        )
                        np.testing.assert_array_equal(
                            np.asarray(p.targets[0])[~active], np.asarray(p.guides[0])[~active]
                        )
                order = METHODS[step % 2 :] + METHODS[: step % 2]
                stats = {n: update(arms[n], prepared[n]) for n in order}
                complete = step + 1
                row = dict(
                    step=complete,
                    depth=depth,
                    scale_level=batch.scale_level,
                    generated=len(records),
                    arms=stats,
                    seconds=time.perf_counter() - begin,
                    clean_sha256=hashlib.sha256(clean.tobytes()).hexdigest(),
                )
                if step < 6 or complete % 25 == 0:
                    with (destination / "training.jsonl").open("a") as f:
                        f.write(json.dumps(row, allow_nan=False) + "\n")
                    write(destination / "progress.json", row)
                    print(json.dumps(row), flush=True)
                if complete % 100 == 0 or complete == 6:
                    lineart_state.save(saved, arms, complete)
                if complete in (6, 1000, STEPS):
                    restore_audit(root, saved, arms, complete)
                    shutil.copytree(
                        destination / f"checkpoints/{complete}",
                        root / f"milestone-checkpoints/{complete}",
                    )
                    if complete != 6:
                        snapshot(root, arms, complete)
                        evaluate(root, arms, complete)
        finally:
            stream.close()
        write(
            destination / f"segment-{start}-{end}.json",
            dict(
                start=start,
                end=end,
                depth_counts=counts,
                initial_pair_equal=True,
                masks_paired=True,
                clean_targets_protected_correct=True,
            ),
        )


def audit_snapshots(root):
    check(root)
    bindings = {}
    for step in (1000, STEPS):
        arms = make_arms(root)
        with checkpoints.manager(root / "milestone-checkpoints", keep=3) as saved:
            lineart_state.restore(saved, arms, step)
        for name, arm in arms.items():
            for mode in ("raw", "ema"):
                expected = nnx.clone(arm["model"])
                if mode == "ema":
                    nnx.update(expected, arm["ema"])
                folder = root / f"snapshots/step-{step}/{name}/{mode}"
                actual, config = load_inference(folder)
                assert config == arm["config"]
                jax.tree.map(
                    np.testing.assert_array_equal,
                    nnx.state(expected, nnx.Param),
                    nnx.state(actual, nnx.Param),
                )
                bindings.update({str(p): sha(p) for p in folder.iterdir() if p.is_file()})
    write(
        root / "snapshot-audit.json",
        dict(full_state_restore_exact=True, raw_ema_exact=True, models=8, bindings=bindings),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "freeze", "train", "audit"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--end", type=int, default=STEPS)
    args = p.parse_args()
    root = args.root.resolve()
    if args.action == "prepare":
        prepare(root)
    elif args.action == "freeze":
        freeze(root)
    elif args.action == "audit":
        audit_snapshots(root)
    else:
        with (root / "process.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            train(root, args.end)
