"""Compare Charbonnier with equal-path profile supervision at fixed capacity."""

import argparse
import fcntl
import json
import os
import shutil
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from campaign_protocol import read, sha, write
from cue_capacity import forward, make_arm, prepare_inputs
from cue_preserving_data import MODES, strength_map
from diagnose_learnability import freeze, snapshot, verify
from flax import nnx
from path_profile_data import training_annotation
from path_profile_objective import profile_gradient
from plausible_study import digest, exact, replay_comparison

from lineartgen.config import save_config
from lineartgen.models.serialization import load_inference, save_inference
from lineartgen.training import checkpoints, lineart_state

METHODS = ("baseline", "profile")
STEPS = 6000
SAVED = (0, 6, STEPS)
DOCUMENT = Path("docs/path-profile-plan-20261013.md")
PARENT = Path("../jax-family-robust-20261013/runs/robust-family-20261013")
SEED = 2026101312
EVAL_SEED = 2026101313


def make_arms(root):
    model, config = load_inference(root / "initial")
    result = {m: make_arm(config, model) for m in METHODS}
    for method, arm in result.items():
        arm["gradient"] = profile_gradient(0.25 if method == "profile" else 0.0)
    return result


def selection(step):
    rng = np.random.default_rng([SEED, step])
    return dict(
        indices=rng.integers(64, size=4).tolist(),
        variants=rng.integers(8, size=4).tolist(),
        modes=rng.integers(len(MODES), size=4).tolist(),
        depth=step % 6 + 1,
    )


def training_batch(data, step, method):
    selected = selection(step)
    indices = np.asarray(selected["indices"]) * 8
    indices += np.asarray(selected["variants"])
    candidates = jnp.asarray(data["variants"][indices])
    source = jnp.asarray(data["source"][indices])
    strength = np.stack([strength_map(MODES[m]) for m in selected["modes"]])
    prepared = prepare_inputs(
        candidates[:, 13],
        source,
        strength,
        jax.random.fold_in(jax.random.PRNGKey(SEED), step),
        depth=selected["depth"],
    )
    return prepared, candidates, {k: jnp.asarray(data[k][indices]) for k in ("probes", "valid")}


def update(arm, prepared, candidates, annotation):
    parts, grads = arm["gradient"](arm["model"], prepared, candidates, annotation)
    parts["gradient_norm"] = optax.global_norm(grads)
    metrics = {k: float(v) for k, v in parts.items()}
    assert all(np.isfinite(x) for x in metrics.values())
    _, arm["ema"] = arm["update"](arm["model"], arm["optimizer"], grads, arm["ema"])
    jax.block_until_ready(arm["ema"])
    return metrics


def prepare(root):
    parent = PARENT.resolve()
    assert read(parent / "completion.json")["stage"] == "complete_robust_family"
    root.mkdir(parents=True, exist_ok=False)
    model, old = load_inference(parent / "snapshots/step-6000/dynamic/raw")
    config = old.model_copy(
        update={
            "train": old.train.model_copy(
                update={"steps": STEPS, "learning_rate_decay_steps": STEPS, "seed": SEED}
            )
        }
    )
    assert config.train.peak_learning_rate == 1e-4
    save_inference(
        model, config, root / "initial", provenance="CX dynamic Charbonnier raw6000 common initial"
    )
    (root / "configs").mkdir()
    for method in METHODS:
        save_config(config, root / f"configs/{method}.json")
    shutil.copytree(parent / "data", root / "data")
    for name in ("train.npz", "data.json"):
        shutil.copy2(parent / name, root / name)
    annotations = [training_annotation(r["metadata"]) for r in read(root / "data.json")["records"]]
    np.savez_compressed(
        root / "path-annotations.npz",
        **{k: np.stack([r[k] for r in annotations]) for k in ("probes", "valid")},
    )
    write(root / "sampling-plan.json", [dict(step=k, **selection(k)) for k in range(STEPS)])
    write(root / "train.json", dict(cases=512, seed=SEED, steps=STEPS, profile_coefficient=0.25))
    write(
        root / "probe.json",
        dict(cases=256, seed=EVAL_SEED, modes=MODES, draws=2, depth=3, confirmation=False),
    )
    bindings = {str(p): sha(p) for p in root.rglob("*") if p.is_file()}
    for p in (
        parent / "completion.json",
        *list((parent / "snapshots/step-6000/dynamic/raw").iterdir()),
    ):
        if p.is_file():
            bindings[str(p)] = sha(p)
    write(
        root / "plan.json",
        dict(
            initial=str(root / "initial"),
            initial_files={p.name: sha(p) for p in (root / "initial").iterdir()},
            data={},
            bound_files=bindings,
            parent=str(parent),
            methods=METHODS,
            steps=STEPS,
            saved=SAVED,
            plan_document_sha256=sha(DOCUMENT),
            adoption=False,
        ),
    )


def check(root):
    verify(root)
    plan = read(root / "plan.json")
    assert sha(DOCUMENT) == plan["plan_document_sha256"]
    for file, h in plan["bound_files"].items():
        assert sha(Path(file)) == h, file
    return plan


def train(root):
    check(root)
    assert os.environ.get("XLA_FLAGS", "") == "" and jax.devices()[0].platform != "cpu"
    assert read(root / "test-results.json")["passed"]
    arms = make_arms(root)
    with np.load(root / "train.npz") as z:
        data = {k: z[k] for k in z.files}
    with np.load(root / "path-annotations.npz") as z:
        data.update({k: z[k] for k in z.files})
    for step in range(6):
        p, _, _ = training_batch(data, step, "baseline")
        exact(forward(arms["baseline"]["model"], p), forward(arms["profile"]["model"], p))
    write(
        root / "initial-audit.json", dict(six_depth_outputs_exact=True, generator_parameters=15809)
    )
    write(
        root / "backend.json",
        dict(flags=os.environ.get("XLA_FLAGS", ""), device=jax.devices()[0].device_kind),
    )
    out = root / "training"
    out.mkdir(exist_ok=True)
    with checkpoints.manager(out / "checkpoints", keep=len(SAVED)) as saved:
        start = lineart_state.restore(saved, arms) if saved.latest_step() is not None else 0
        journal = out / "sampling.jsonl"
        if journal.exists():
            records = [json.loads(s) for s in journal.read_text().splitlines()]
            records = [x for x in records if x["step"] <= start]
            assert [x["step"] for x in records] == list(range(1, start + 1))
            journal.write_text("".join(json.dumps(x) + "\n" for x in records))
        if start == 0:
            lineart_state.save(saved, arms, 0)
            snapshot(root, arms, 0)
        prefix = []
        for step in range(start, STEPS):
            stats, elapsed, inputs = {}, {}, {}
            for name in METHODS[step % 2 :] + METHODS[: step % 2]:
                p, candidates, annotation = training_batch(data, step, name)
                inputs[name] = digest((p, candidates, annotation))
                tick = time.perf_counter()
                stats[name] = update(arms[name], p, candidates, annotation)
                elapsed[name] = time.perf_counter() - tick
            row = dict(
                step=step + 1,
                **selection(step),
                metrics=stats,
                seconds=elapsed,
                input_sha256=digest(tuple(inputs[m] for m in METHODS)),
                per_arm_input_sha256=inputs,
            )
            with journal.open("a") as f:
                f.write(json.dumps(row) + "\n")
            if step < 6:
                prefix.append(stats)
            if step < 6 or (step + 1) % 100 == 0:
                write(out / "progress.json", dict(step=step + 1, metrics=stats))
                print(dict(step=step + 1), flush=True)
            if step == 5:
                replay = make_arms(root)
                metric_audit = []
                for k in range(6):
                    actual = {m: update(replay[m], *training_batch(data, k, m)) for m in METHODS}
                    metric_audit.append(
                        {m: replay_comparison(actual[m], prefix[k][m], "prior") for m in METHODS}
                    )
                expected = lineart_state.payload(arms, 6)["arms"]
                actual = lineart_state.payload(replay, 6)["arms"]
                write(
                    root / "prefix-audit.json",
                    dict(
                        passed=True,
                        metrics=metric_audit,
                        states={
                            m: replay_comparison(actual[m], expected[m], "prior") for m in METHODS
                        },
                    ),
                )
            if step + 1 in SAVED:
                lineart_state.save(saved, arms, step + 1)
                restored = make_arms(root)
                assert lineart_state.restore(saved, restored, step + 1) == step + 1
                exact(
                    lineart_state.payload(arms, step + 1), lineart_state.payload(restored, step + 1)
                )
                snapshot(root, arms, step + 1)
                write(out / f"restore-{step + 1}.json", dict(full_state_exact=True))
        bindings = {}
        for step in SAVED:
            restored = make_arms(root)
            assert lineart_state.restore(saved, restored, step) == step
            for name, arm in restored.items():
                for mode in ("raw", "ema"):
                    expected = nnx.clone(arm["model"])
                    if mode == "ema":
                        nnx.update(expected, arm["ema"])
                    folder = root / f"snapshots/step-{step}/{name}/{mode}"
                    actual, config = load_inference(folder)
                    assert config == arm["config"]
                    exact(nnx.state(expected, nnx.Param), nnx.state(actual, nnx.Param))
                    bindings.update({str(p): sha(p) for p in folder.iterdir() if p.is_file()})
        write(root / "snapshot-audit.json", dict(raw_ema_exact=True, exports=12, bindings=bindings))
    write(root / "workflow-status.json", dict(stage="training_complete", step=STEPS))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "freeze", "train"))
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    if a.action == "train":
        with (root / "process.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            write(root / "workflow-status.json", dict(stage="training"))
            train(root)
    elif a.action == "freeze":
        freeze(root)
    else:
        prepare(root)
