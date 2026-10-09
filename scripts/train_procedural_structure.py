"""Shared online geometry study: baseline, local region, region plus connection."""

import argparse
import fcntl
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
from procedural_queries import generate, query_loss
from train_lineart_loss_study import restore_journal

from lineartgen.config import load_config, save_config
from lineartgen.diffusion import recursive_forward_with_preclip, training_inputs
from lineartgen.training import checkpoints, lineart_state
from lineartgen.training.lineart import line_region_loss
from lineartgen.training.losses import image_loss

SEED = 2026100951
STEPS = 4000
METHODS = {"pixel": (0.0, 0.0), "region": (0.1, 0.0), "structure": (0.1, 0.1)}


def prepare(root, parent):
    assert not root.exists()
    (root / "configs").mkdir(parents=True)
    plan = read(parent / "plan.json")
    base = load_config(Path(plan["initial"]) / "config.json")
    base = base.model_copy(
        update={
            "train": base.train.model_copy(
                update={
                    "steps": STEPS,
                    "batch_size": 4,
                    "accumulate_steps": 1,
                    "learning_rate": 1e-4,
                    "minimum_learning_rate": 1e-5,
                    "warmup_steps": 100,
                    "seed": SEED,
                }
            ),
            "sketch": base.sketch.model_copy(
                update={
                    "mode": "blur",
                    "probability": 0.8,
                    "opacity_min": 0.25,
                    "opacity_max": 1.0,
                    "blur_sigma_min": 0.0,
                    "blur_sigma_max": 2.0,
                }
            ),
        }
    )
    assert base.train.peak_learning_rate == 2e-4
    assert base.diffusion.scale_loss_multiplier == 0 and base.matching.method == "independent"
    for name in METHODS:
        save_config(base, root / f"configs/{name}.json")
    for n in ("train.json", "probe.json", "inputs.npz", "queries.json"):
        shutil.copyfile(parent / n, root / n)
    write(
        root / "plan.json",
        dict(
            arms=list(METHODS),
            warm_start_methods=list(METHODS),
            coefficients=METHODS,
            initial=plan["initial"],
            initial_files=plan["initial_files"],
            data=plan["data"],
            parent=str(parent),
            seed=SEED,
            steps=STEPS,
            milestones=[0, 1000, STEPS],
            plan_document_sha256=sha(Path("docs/procedural-structure-plan-20261009.md")),
            inputs_sha256=sha(root / "inputs.npz"),
            queries_sha256=sha(root / "queries.json"),
            diagnostic_only=True,
            adoption=False,
            sealed_confirmation_generated=False,
        ),
    )


def make_arms(root):
    arms = arms_from_root(root)
    for name, arm in arms.items():
        config = arm["config"]
        region_weight, connection_weight = METHODS[name]

        def objective(model, batch, config=config, rw=region_weight, cw=connection_weight):
            prepared, queries = batch
            predictions, physical = recursive_forward_with_preclip(
                model, prepared.noisy, prepared.noise_levels, guides=prepared.guides
            )
            assert config.diffusion.scale_loss_multiplier == 0
            prediction, target, noise = (
                predictions[0],
                prepared.targets[0],
                prepared.noise_levels[0],
            )
            base, metrics = image_loss(
                prediction,
                target,
                noise,
                config.loss,
                config.diffusion.min_noise,
                preclip=physical[0],
            )
            active = noise > config.diffusion.min_noise
            region = line_region_loss(prediction, target, active) if rw else jnp.zeros(())
            connection, extra = (
                query_loss(prediction, target, active, queries) if cw else (jnp.zeros(()), {})
            )
            total = base + rw * region + cw * connection
            return total, dict(
                metrics,
                **extra,
                loss=total,
                base_loss=base,
                region_loss=region,
                connection_loss=connection,
            )

        # Bind the objective now; each arm must keep its own coefficients.
        def make_gradient(objective):
            @nnx.jit
            def gradient(model, prepared):
                (_, metrics), gradients = nnx.value_and_grad(objective, has_aux=True)(
                    model, prepared
                )
                return metrics, gradients

            return gradient

        arm["gradient"] = make_gradient(objective)
    return arms


def prepare_batch(config):
    @jax.jit(static_argnums=2)
    def prepare(clean, key, levels):
        return training_inputs(
            clean,
            key,
            config.diffusion,
            config.matching,
            levels=levels,
            sketch=config.sketch,
            with_source=True,
        )

    return prepare


def evaluate(root, arms, step):
    # Reuse the locked64-condition diagnostic; broaden all57 pairs separately.
    import train_connection_queries as comparison

    comparison.evaluate(root, arms, step)


def train(root, preflight, end):
    verify(root)
    assert jax.devices()[0].platform != "cpu" and end in (1000, STEPS)
    if not preflight:
        assert read(root / "preflight/restore-audit.json")["all_state_exact"]
    arms = make_arms(root)
    prepare = prepare_batch(arms["pixel"]["config"])
    destination = root / ("preflight" if preflight else "training")
    destination.mkdir(exist_ok=True)
    with checkpoints.manager(destination / "checkpoints", keep=3) as saved:
        start = lineart_state.restore(saved, arms) if saved.latest_step() is not None else 0
        restore_journal(destination, start)
        if start == 0 and not preflight:
            snapshot(root, arms, 0)
            evaluate(root, arms, 0)
        for step in range(start, 6 if preflight else end):
            begin = time.perf_counter()
            clean, queries, records = generate(SEED, step)
            prepared = prepare(
                jnp.asarray(clean), jax.random.fold_in(jax.random.PRNGKey(SEED), step), step % 6 + 1
            )
            batch = prepared, jax.tree.map(jnp.asarray, queries)
            names = tuple(arms)
            stats = {
                name: update(arms[name], batch) for name in names[step % 3 :] + names[: step % 3]
            }
            complete = step + 1
            row = dict(
                step=complete,
                depth=step % 6 + 1,
                arms=stats,
                records=records,
                seconds=time.perf_counter() - begin,
            )
            if preflight or complete % 25 == 0:
                with (destination / "training.jsonl").open("a") as f:
                    f.write(json.dumps(row, allow_nan=False) + "\n")
                write(destination / "progress.json", row)
                print(json.dumps(row), flush=True)
            if complete % 100 == 0 or complete == 6:
                lineart_state.save(saved, arms, complete)
                if complete in (1000, STEPS):
                    shutil.copytree(
                        destination / f"checkpoints/{complete}",
                        root / f"milestone-checkpoints/{complete}",
                    )
            if not preflight and complete == 6:
                check = make_arms(root)
                with checkpoints.manager(root / "preflight/checkpoints", keep=3) as previous:
                    lineart_state.restore(previous, check, 6)
                jax.tree.map(
                    np.testing.assert_array_equal,
                    lineart_state.payload(arms, 6),
                    lineart_state.payload(check, 6),
                )
                write(root / "replay-audit.json", dict(all_state_exact=True, step=6))
            if not preflight and complete in (1000, STEPS):
                snapshot(root, arms, complete)
                evaluate(root, arms, complete)
        restored = make_arms(root)
        last = lineart_state.restore(saved, restored)
        jax.tree.map(
            np.testing.assert_array_equal,
            lineart_state.payload(arms, last),
            lineart_state.payload(restored, last),
        )
        write(destination / "restore-audit.json", dict(all_state_exact=True, step=last))
        write(destination / f"restore-audit-{last}.json", dict(all_state_exact=True, step=last))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "freeze", "preflight", "train"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--end", type=int, default=STEPS)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "prepare":
        prepare(root, args.parent.resolve())
    elif args.action == "freeze":
        freeze(root)
    else:
        with (root / "process.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            train(root, args.action == "preflight", args.end)
