"""Local opacity transport on deterministic original-rough features."""

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
from diagnose_learnability import freeze, snapshot, update, verify
from procedural_queries import generate
from train_connection_queries import evaluate as evaluate_primary
from train_lineart_loss_study import restore_journal
from train_local_transport import make_arms as transport_arms
from train_procedural_structure import SEED, STEPS, prepare_batch
from train_source_signal import make_arms as parent_arms

from lineartgen.config import load_config, save_config
from lineartgen.models.serialization import load_inference
from lineartgen.training import checkpoints, lineart_state


def prepare(root, parent):
    assert not root.exists()
    verify(parent)
    (root / "configs").mkdir(parents=True)
    old = read(parent / "plan.json")
    for name in ("train.json", "probe.json", "inputs.npz", "queries.json"):
        shutil.copyfile(parent / name, root / name)
    config = load_config(parent / "configs/signal.json")
    assert config.model.feature_input == "source" and config.model.residual_base == "source"
    for name in ("transport",):
        save_config(config, root / f"configs/{name}.json")
    write(
        root / "plan.json",
        dict(
            arms=["transport"],
            warm_start_methods=["transport"],
            comparison_methods=["pixel", "transport"],
            initial=old["initial"],
            initial_files=old["initial_files"],
            data=old["data"],
            parent=str(parent),
            steps=STEPS,
            seed=SEED,
            milestones=[0, 1000, STEPS],
            coefficient=3.0,
            tile=16,
            distance="smooth_absolute",
            intervention="Source features plus local opacity transport",
            plan_document_sha256=sha(Path("docs/signal-transport-plan-20261009.md")),
            parent_files={
                n: sha(parent / n)
                for n in (
                    "source.json",
                    "plan.json",
                    "configs/signal.json",
                    "preflight/restore-audit.json",
                    "replay-audit.json",
                )
            },
            inputs_sha256=sha(root / "inputs.npz"),
            queries_sha256=sha(root / "queries.json"),
            diagnostic_only=True,
            adoption=False,
            sealed_confirmation_generated=False,
        ),
    )


def check(root):
    verify(root)
    plan = read(root / "plan.json")
    parent = Path(plan["parent"])
    verify(parent)
    assert all(sha(parent / n) == d for n, d in plan["parent_files"].items())
    assert sha(root / "inputs.npz") == plan["inputs_sha256"]
    assert sha(root / "queries.json") == plan["queries_sha256"]
    if (root / "parent-final-binding.json").exists():
        assert all(
            sha(parent / n) == d
            for n, d in read(root / "parent-final-binding.json")["files"].items()
        )
    return plan


def bind_parent(root):
    plan = check(root)
    parent = Path(plan["parent"])
    assert read(parent / "training/restore-audit-4000.json")["all_state_exact"]
    files = [parent / "training/restore-audit-4000.json", parent / "broad/step-4000/metrics.json"]
    for step in (0, 1000, STEPS):
        source = parent / f"snapshots/step-{step}/signal"
        destination = root / f"snapshots/step-{step}/pixel"
        if not destination.exists():
            shutil.copytree(source, destination)
        files += [p for p in source.rglob("*") if p.is_file()]
        for p in source.rglob("*"):
            if p.is_file():
                assert sha(p) == sha(destination / p.relative_to(source))
        files += [parent / f"evaluation/step-{step}/predictions.npz"]
    write(
        root / "parent-final-binding.json",
        dict(
            files={str(p.relative_to(parent)): sha(p) for p in files},
            copied_baseline_models_exact=True,
            predeclared_endpoint=STEPS,
        ),
    )


def make_arms(root):
    # Reuse the exact prior local-transport objective and coefficient.
    return transport_arms(root)


def replay(root):
    parent = Path(check(root)["parent"])
    arms = parent_arms(parent)
    control = arms["signal"]
    prepare = prepare_batch(control["config"])
    for step in range(6):
        clean, queries, _ = generate(SEED, step)
        p = prepare(
            jnp.asarray(clean), jax.random.fold_in(jax.random.PRNGKey(SEED), step), step % 6 + 1
        )
        update(control, (p, jax.tree.map(jnp.asarray, queries)))
    expected = parent_arms(parent)
    with checkpoints.manager(parent / "preflight/checkpoints", keep=3) as saved:
        lineart_state.restore(saved, expected, 6)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload({"pixel": control}, 6),
        lineart_state.payload({"pixel": expected["signal"]}, 6),
    )
    write(root / "control-replay.json", dict(all_state_exact=True, steps=6))
    print("baseline six-update full-state replay exact", flush=True)


def evaluate(root, arms, step):
    plan = check(root)
    control, _ = load_inference(root / f"snapshots/step-{step}/pixel/raw")
    evaluate_primary(root, dict(pixel=dict(model=control), **arms), step)
    actual = np.load(root / f"evaluation/step-{step}/predictions.npz")
    expected = np.load(Path(plan["parent"]) / f"evaluation/step-{step}/predictions.npz")
    keys = [key for key in actual.files if key.startswith("pixel-")]
    assert len(keys) == 64
    for key in keys:
        np.testing.assert_array_equal(actual[key], expected[key.replace("pixel-", "signal-", 1)])
    write(root / f"evaluation/step-{step}/control-audit.json", dict(all_exact=True, conditions=64))


def audit_broad(root, step):
    parent = Path(check(root)["parent"])
    for split in ("train", "probe"):
        actual = np.load(root / f"broad/step-{step}/pixel-{split}.npz")["images"]
        expected = np.load(parent / f"broad/step-{step}/signal-{split}.npz")["images"]
        np.testing.assert_array_equal(actual, expected)
    actual = [
        r for r in read(root / f"broad/step-{step}/metrics.json")["rows"] if r["method"] == "pixel"
    ]
    expected = [
        dict(r, method="pixel")
        for r in read(parent / f"broad/step-{step}/metrics.json")["rows"]
        if r["method"] == "signal"
    ]
    assert actual == expected and len(actual) == 456
    write(root / f"broad/step-{step}/control-audit.json", dict(all_exact=True, conditions=456))
    print("all456 broad control predictions and metrics exact", flush=True)


def train(root, preflight, end):
    check(root)
    assert end in (1000, STEPS) and jax.devices()[0].platform != "cpu"
    assert read(root / "control-replay.json")["all_state_exact"]
    if not preflight:
        assert read(root / "preflight/restore-audit.json")["all_state_exact"]
    arms = make_arms(root)
    prepare = prepare_batch(arms["transport"]["config"])
    destination = root / ("preflight" if preflight else "training")
    destination.mkdir(exist_ok=True)
    with checkpoints.manager(destination / "checkpoints", keep=3) as saved:
        start = lineart_state.restore(saved, arms) if saved.latest_step() is not None else 0
        restore_journal(destination, start)
        if start == 0:
            snapshot(root, arms, 0)
            evaluate(root, arms, 0)
        for step in range(start, 6 if preflight else end):
            begin = time.perf_counter()
            clean, queries, records = generate(SEED, step)
            p = prepare(
                jnp.asarray(clean), jax.random.fold_in(jax.random.PRNGKey(SEED), step), step % 6 + 1
            )
            batch = (p, jax.tree.map(jnp.asarray, queries))
            names = tuple(arms)
            metrics = {
                name: update(arms[name], batch)
                for name in names[step % len(names) :] + names[: step % len(names)]
            }
            complete = step + 1
            row = dict(
                step=complete,
                depth=step % 6 + 1,
                arms=metrics,
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
                expected = make_arms(root)
                with checkpoints.manager(root / "preflight/checkpoints", keep=3) as previous:
                    lineart_state.restore(previous, expected, 6)
                jax.tree.map(
                    np.testing.assert_array_equal,
                    lineart_state.payload(arms, 6),
                    lineart_state.payload(expected, 6),
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
    parser.add_argument(
        "action", choices=("prepare", "freeze", "bind", "replay", "preflight", "train", "audit")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--end", type=int, default=STEPS)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "prepare":
        prepare(root, args.parent.resolve())
    elif args.action == "freeze":
        freeze(root)
    elif args.action == "bind":
        bind_parent(root)
    elif args.action == "audit":
        audit_broad(root, args.end)
    else:
        with (root / "process.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.action == "replay":
                replay(root)
            else:
                train(root, args.action == "preflight", args.end)
