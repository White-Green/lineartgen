"""Matched width16/32 diagnostic with fixed cue-preserving synthetic pairs."""

import argparse
import fcntl
import hashlib
import json
import os
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from campaign_protocol import read, sha, write
from cue_preserving_data import COUNT, MODES, SIZE, make_case, strength_map, transfer
from diagnose_learnability import freeze, snapshot, update, verify
from flax import nnx
from local_recursive_fit import fresh_model, local_config

from lineartgen.config import LossConfig, load_config, save_config
from lineartgen.diffusion import ProgressTrainingInputs, pyramid, recursive_forward_with_preclip
from lineartgen.models.serialization import load_inference, save_inference
from lineartgen.training import checkpoints, lineart_state
from lineartgen.training.steps import create_optimizer, make_steps

STEPS = 6000
SAVED = (0, 6, 2000, STEPS)
METHODS = ("width16", "width32")
PARAMETERS = {"width16": 15809, "width32": 59265}
SEED = 2026101292
NOISE_SEED = 2026101293
EVAL_SEED = 2026101294
PLAN_DOCUMENT = Path("docs/cue-preserving-capacity-plan-20261012.md")


@partial(jax.jit, static_argnames=("depth",))
def prepare_inputs(clean, source, strength, key, depth):
    guides, levels, targets = (
        pyramid(source, depth),
        pyramid(strength, depth),
        pyramid(clean, depth),
    )
    keys = jax.random.split(key, depth)
    noisy = tuple(
        s + 0.25 * t * jax.random.normal(k, s.shape)
        for s, t, k in zip(guides, levels, keys, strict=True)
    )
    targets = tuple(jnp.where(t > 0, y, s) for y, t, s in zip(targets, levels, guides, strict=True))
    return ProgressTrainingInputs(
        targets, levels, noisy, guides, tuple(jnp.ones_like(s) for s in guides)
    )


@nnx.jit
def forward(model, prepared):
    return recursive_forward_with_preclip(
        model,
        prepared.noisy,
        prepared.noise_levels,
        guides=prepared.guides,
        progress=prepared.progress,
    )


def make_arm(config, model):
    model = nnx.clone(model)
    optimizer, schedule = create_optimizer(model, config)
    gradient, updater, measure = make_steps(config)
    return dict(
        config=config,
        model=model,
        optimizer=optimizer,
        schedule=schedule,
        gradient=gradient,
        update=updater,
        measure=measure,
        critic=None,
        ema=jax.tree.map(lambda x: x.copy(), nnx.state(model, nnx.Param)),
    )


def make_arms(root):
    arms = {}
    for name in METHODS:
        m, c = load_inference(root / f"initials/{name}")
        assert c == load_config(root / f"configs/{name}.json")
        arms[name] = make_arm(c, m)
    return arms


def digest_tree(value):
    return hashlib.sha256(
        b"".join(np.asarray(x).tobytes() for x in jax.tree.leaves(value))
    ).hexdigest()


def select_batch(step):
    rng = np.random.default_rng([SEED, step])
    return rng.integers(COUNT, size=4).tolist(), rng.integers(len(MODES), size=4).tolist()


def training_batch(data, step):
    indices, modes = select_batch(step)
    strength = np.stack([strength_map(MODES[m]) for m in modes])
    return prepare_inputs(
        jnp.asarray(data["clean"][indices]),
        jnp.asarray(data["source"][indices]),
        jnp.asarray(strength),
        jax.random.fold_in(jax.random.PRNGKey(NOISE_SEED), step),
        step % 6 + 1,
    )


def prepare(root):
    assert not root.exists()
    base = Path(
        "../jax-local-zero-noise-20261012/runs/local-zero-noise-20261012/configs/local.json"
    ).resolve()
    old = load_config(base)
    train = old.train.model_copy(
        update=dict(
            steps=STEPS,
            learning_rate_decay_steps=STEPS,
            learning_rate=5e-5,
            minimum_learning_rate=1e-5,
            warmup_steps=0,
            model_seed=42,
            seed=SEED,
            batch_size=4,
            accumulate_steps=1,
        )
    )
    loss = LossConfig(
        pixel_loss="interval_l1",
        balance_tones=True,
        tone_balance_power=0.5,
        noise_weight_floor=0.05,
        identity_weight=0.1,
        edge_weight=0,
        contrast_weight=0,
    )
    root.mkdir(parents=True)
    (root / "configs").mkdir()
    for name, width in zip(METHODS, (16, 32), strict=True):
        c = old.model_copy(
            update=dict(
                model=local_config().model_copy(update={"channels": (width,)}),
                train=train,
                loss=loss,
                data=old.data.model_copy(update={"crop_size": SIZE}),
                diffusion=old.diffusion.model_copy(update={"scale_loss_multiplier": 0.5}),
            )
        )
        assert c.train.peak_learning_rate == 1e-4
        save_config(c, root / f"configs/{name}.json")
        m = fresh_model(c)
        assert sum(x.size for x in jax.tree.leaves(nnx.state(m, nnx.Param))) == PARAMETERS[name]
        save_inference(
            m,
            c,
            root / f"initials/{name}",
            provenance="Fresh seed42 zero-head cue capacity diagnostic",
        )
    records, bound = [], {str(base): sha(base)}
    hashes = {}
    for split in ("train", "probe"):
        arrays = []
        for index in range(COUNT):
            data, record = make_case(split, index)
            p = root / f"data/{split}/{index:03d}.npz"
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, **data)
            record.update(file=str(p.relative_to(root)), sha256=sha(p))
            records.append(record)
            arrays.append(data)
        p = root / f"{split}.npz"
        np.savez_compressed(
            p,
            clean=np.stack([a["clean"] for a in arrays]),
            source=np.stack([a["source"] for a in arrays]),
        )
        bound[str(p)] = sha(p)
        hashes[split] = {
            r["clean_sha256"]
            for r in records
            if r["split"] == split and r["kind"] not in ("blank", "black")
        }
    assert hashes["train"].isdisjoint(hashes["probe"])
    assert max(r["inverse_max_error"] for r in records) < 2e-6
    write(
        root / "data.json",
        dict(
            records=records,
            nonconstant_train_probe_disjoint=True,
            spectral_min=float(transfer((SIZE, SIZE)).min()),
            spectral_max=float(transfer((SIZE, SIZE)).max()),
            inverse_max_error=max(r["inverse_max_error"] for r in records),
        ),
    )
    bound[str(root / "data.json")] = sha(root / "data.json")
    bound.update({str(p): sha(p) for p in (root / "initials").rglob("*") if p.is_file()})
    write(root / "train.json", dict(cases=COUNT, steps=STEPS, seed=SEED, noise_seed=NOISE_SEED))
    write(
        root / "probe.json",
        dict(
            cases=COUNT,
            noise_seed=EVAL_SEED,
            draws=2,
            modes=list(MODES),
            depth=3,
            confirmation=False,
        ),
    )
    audit = [
        dict(step=i, indices=select_batch(i)[0], modes=select_batch(i)[1], depth=i % 6 + 1)
        for i in range(STEPS)
    ]
    write(root / "sampling-plan.json", audit)
    bound[str(root / "sampling-plan.json")] = sha(root / "sampling-plan.json")
    write(
        root / "plan.json",
        dict(
            initial=str(root / "initials/width16"),
            initial_files={p.name: sha(p) for p in (root / "initials/width16").iterdir()},
            data={},
            bound_files=bound,
            steps=STEPS,
            saved=list(SAVED),
            methods=list(METHODS),
            parameters=PARAMETERS,
            plan_document_sha256=sha(PLAN_DOCUMENT),
            adoption=False,
        ),
    )


def check(root):
    verify(root)
    plan = read(root / "plan.json")
    assert sha(PLAN_DOCUMENT) == plan["plan_document_sha256"]
    for path, digest in plan["bound_files"].items():
        assert sha(Path(path)) == digest, path
    for r in read(root / "data.json")["records"]:
        assert sha(root / r["file"]) == r["sha256"]
    return plan


def exact(a, b):
    jax.tree.map(np.testing.assert_array_equal, a, b)


def train(root):
    check(root)
    assert os.environ.get("XLA_FLAGS", "") == "" and jax.devices()[0].platform != "cpu"
    assert read(root / "test-results.json")["passed"]
    write(
        root / "backend.json",
        dict(flags=os.environ.get("XLA_FLAGS", ""), device=jax.devices()[0].device_kind),
    )
    arms = make_arms(root)
    with np.load(root / "train.npz") as stored:
        data = {k: stored[k] for k in stored.files}
    initial = []
    for depth in range(1, 7):
        p = training_batch(data, depth - 1)
        values = [forward(a["model"], p) for a in arms.values()]
        exact(*values)
        losses = [a["measure"](a["model"], p) for a in arms.values()]
        exact(*losses)
        initial.append(digest_tree(values[0]))
    write(
        root / "initial-audit.json", dict(six_depth_outputs_and_losses_exact=True, digests=initial)
    )
    out = root / "training"
    out.mkdir(exist_ok=True)
    with checkpoints.manager(out / "checkpoints", keep=len(SAVED)) as saved:
        start = lineart_state.restore(saved, arms) if saved.latest_step() is not None else 0
        journal = out / "sampling.jsonl"
        if journal.exists():
            rows = [json.loads(s) for s in journal.read_text().splitlines()]
            rows = [r for r in rows if r["step"] <= start]
            journal.write_text("".join(json.dumps(r) + "\n" for r in rows))
            assert [r["step"] for r in rows] == list(range(1, start + 1))
        if start == 0:
            lineart_state.save(saved, arms, 0)
            snapshot(root, arms, 0)
        prefix = []
        for step in range(start, STEPS):
            p = training_batch(data, step)
            stats, times = {}, {}
            for name in METHODS[:: 1 if step % 2 == 0 else -1]:
                tick = time.perf_counter()
                stats[name] = update(arms[name], p)
                times[name] = time.perf_counter() - tick
            row = dict(
                step=step + 1,
                depth=step % 6 + 1,
                indices=select_batch(step)[0],
                modes=select_batch(step)[1],
                input_sha256=digest_tree(p),
                metrics=stats,
                seconds=times,
            )
            with journal.open("a") as f:
                f.write(json.dumps(row) + "\n")
            if step < 6:
                prefix.append(stats)
            if step < 6 or (step + 1) % 100 == 0:
                write(
                    out / "progress.json",
                    dict(step=step + 1, loss={m: stats[m]["loss"] for m in METHODS}),
                )
                print(step + 1, {m: stats[m]["loss"] for m in METHODS}, flush=True)
            if step + 1 == 6:
                replay = make_arms(root)
                for k in range(6):
                    q = training_batch(data, k)
                    actual = {m: update(replay[m], q) for m in METHODS}
                    if start == 0:
                        assert actual == prefix[k]
                exact(lineart_state.payload(arms, 6), lineart_state.payload(replay, 6))
                write(
                    root / "prefix-audit.json", dict(all_state_exact=True, metrics_exact=start == 0)
                )
            if step + 1 in SAVED:
                lineart_state.save(saved, arms, step + 1)
                restored = make_arms(root)
                assert lineart_state.restore(saved, restored, step + 1) == step + 1
                exact(
                    lineart_state.payload(arms, step + 1), lineart_state.payload(restored, step + 1)
                )
                snapshot(root, arms, step + 1)
                write(out / f"restore-{step + 1}.json", dict(all_state_exact=True))
        for step in SAVED:
            restored = make_arms(root)
            assert lineart_state.restore(saved, restored, step) == step
            for name, arm in restored.items():
                for mode in ("raw", "ema"):
                    expected = nnx.clone(arm["model"])
                    if mode == "ema":
                        nnx.update(expected, arm["ema"])
                    actual, c = load_inference(root / f"snapshots/step-{step}/{name}/{mode}")
                    assert c == arm["config"]
                    exact(nnx.state(expected, nnx.Param), nnx.state(actual, nnx.Param))
        write(root / "snapshot-audit.json", dict(raw_ema_exact=True, exports=16))
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
