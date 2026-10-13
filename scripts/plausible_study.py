"""Matched 16ch generator updates for pixel, tolerant geometry, and geometry+GAN."""

import argparse
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from campaign_protocol import read, sha, write
from cue_capacity import forward, prepare_inputs
from cue_capacity import make_arm as pixel_arm
from cue_preserving_data import MODES, strength_map
from diagnose_learnability import freeze, snapshot, verify
from diagnose_learnability import update as pixel_update
from flax import nnx
from plausible_data import CHOICES, COUNT, make_case
from plausible_objective import PRIOR_WEIGHT, RAMP, create_prior, critic_update, generator_gradient

from lineartgen.config import save_config
from lineartgen.models.serialization import load_inference, save_inference
from lineartgen.training import checkpoints, lineart_state

METHODS = ("pixel", "tolerant", "prior")
STEPS = 4000
SAVED = (0, 6, 1000, STEPS)
SEED = 2026101302
NOISE_SEED = 2026101304
EVAL_SEED = 2026101305
DOCUMENT = Path("docs/plausible-lineart-replay-plan-20261013.md")


def digest(tree):
    return hashlib.sha256(
        b"".join(np.asarray(x).tobytes() for x in jax.tree.leaves(tree))
    ).hexdigest()


def make_arm(c, model, name):
    arm = pixel_arm(c, model)
    if name != "pixel":
        arm["gradient"] = generator_gradient(adversarial=name == "prior")
    if name == "prior":
        arm["critic"], arm["critic_optimizer"] = create_prior()
    return arm


def make_arms(root):
    model, c = load_inference(root / "initial")
    return {m: make_arm(c, model, m) for m in METHODS}


def selection(step):
    rng = np.random.default_rng([SEED, step])
    return dict(
        indices=rng.integers(COUNT, size=4).tolist(),
        choices=rng.integers(len(CHOICES), size=4).tolist(),
        modes=rng.integers(len(MODES), size=4).tolist(),
        depth=step % 6 + 1,
    )


def coordinates(prepared, step):
    result = []
    for level, (guide, t) in enumerate(zip(prepared.guides, prepared.noise_levels, strict=True)):
        guide, t = np.asarray(guide), np.asarray(t)
        size = min(64, guide.shape[1])
        rows = []
        for item in range(len(guide)):
            rng = np.random.default_rng([SEED, step, level, item, 0x434F4F52])
            active = t[item, ..., 0] > 0
            ink = np.argwhere(active & (guide[item, ..., 0] < 0.98))
            paper = np.argwhere(active & (guide[item, ..., 0] > 0.999))
            all_active = np.argwhere(active)
            for points in (ink, ink, paper, all_active):
                if not len(points):
                    points = all_active
                center = points[rng.integers(len(points))] if len(points) else np.array([0, 0])
                rows.append(np.clip(center - size // 2, 0, guide.shape[1] - size))
        result.append(jnp.asarray(np.asarray(rows, np.int32).reshape(len(guide), 4, 2)))
    return tuple(result)


def batch(data, step):
    selected = selection(step)
    indices = selected["indices"]
    clean = data["variants"][indices, selected["choices"]]
    strength = np.stack([strength_map(MODES[m]) for m in selected["modes"]])
    p = prepare_inputs(
        clean,
        data["source"][indices],
        strength,
        jax.random.fold_in(jax.random.PRNGKey(NOISE_SEED), step),
        depth=selected["depth"],
    )
    return p, coordinates(p, step)


def update(arm, name, p, coords, step):
    if name == "pixel":
        return pixel_update(arm, p)
    critic_metrics = {}
    weight = 0.0
    if name == "prior":
        fake = forward(arm["model"], p)[0]
        critic_metrics = critic_update(arm["critic"], arm["critic_optimizer"], fake, p, coords)
        weight = PRIOR_WEIGHT * min((step + 1) / RAMP, 1)
    parts, grads = arm["gradient"](arm["model"], p, coords, arm["critic"], jnp.asarray(weight))
    parts |= (
        dict(gradient_norm=optax.global_norm(grads), prior_weight=jnp.asarray(weight))
        | critic_metrics
    )
    result = {k: float(v) for k, v in parts.items()}
    assert all(np.isfinite(x) for x in result.values()), result
    _, arm["ema"] = arm["update"](arm["model"], arm["optimizer"], grads, arm["ema"])
    jax.block_until_ready(arm["ema"])
    return result


def exact(a, b):
    jax.tree.map(np.testing.assert_array_equal, a, b)


def replay_comparison(actual, expected, name):
    """GPU replay permits tiny floating variation only for the GAN arm."""
    aa, structure = jax.tree.flatten(actual)
    bb, other = jax.tree.flatten(expected)
    assert structure == other
    atol, rtol = (1e-7, 1e-5) if name == "prior" else (0.0, 0.0)
    count, maximum = 0, 0.0
    for a, b in zip(aa, bb, strict=True):
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape and a.dtype == b.dtype
        if np.issubdtype(a.dtype, np.inexact):
            np.testing.assert_allclose(a, b, atol=atol, rtol=rtol, equal_nan=False)
            maximum = max(maximum, float(np.max(np.abs(a - b), initial=0)))
        else:
            np.testing.assert_array_equal(a, b)
        count += int(not np.array_equal(a, b))
    return dict(
        passed=True,
        exact=count == 0,
        differing_leaves=count,
        max_abs_difference=maximum,
        atol=atol,
        rtol=rtol,
    )


def prepare(root):
    assert not root.exists()
    parent = Path(
        "../jax-cue-preserving-capacity-20261012/runs/cue-capacity-20261012/snapshots/step-6000/width16/raw"
    ).resolve()
    model, old = load_inference(parent)
    c = old.model_copy(
        update={
            "train": old.train.model_copy(
                update={"steps": STEPS, "learning_rate_decay_steps": STEPS, "seed": SEED}
            )
        }
    )
    assert c.train.peak_learning_rate == 1e-4 and c.train.accumulate_steps == 1
    assert c.diffusion.scale_loss_multiplier == 0.5 and c.model.channels == (16,)
    root.mkdir(parents=True)
    (root / "configs").mkdir()
    for name in METHODS:
        save_config(c, root / f"configs/{name}.json")
    save_inference(
        model,
        c,
        root / "initial",
        provenance="Common CR width16 raw6000; fresh optimizers, new ambiguity task",
    )
    bound = {str(p): sha(p) for p in parent.iterdir() if p.is_file()}
    records = []
    hashes = {}
    for split in ("train", "probe"):
        cases = []
        for index in range(COUNT):
            data, record = make_case(split, index)
            file = root / f"data/{split}/{index:03d}.npz"
            file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(file, **data)
            records.append(record | dict(file=str(file.relative_to(root)), sha256=sha(file)))
            cases.append(data)
        file = root / f"{split}.npz"
        np.savez_compressed(
            file,
            variants=np.stack([x["variants"] for x in cases]),
            source=np.stack([x["source"] for x in cases]),
        )
        bound[str(file)] = sha(file)
        hashes[split] = {
            r["variants_sha256"]
            for r in records
            if r["split"] == split and r["kind"] not in ("blank", "black")
        }
    assert len(hashes["train"]) == len(hashes["probe"]) == 60 and hashes["train"].isdisjoint(
        hashes["probe"]
    )
    write(root / "data.json", dict(records=records, nonconstant_disjoint=True, choices=CHOICES))
    bound[str(root / "data.json")] = sha(root / "data.json")
    sampling = [dict(step=step, **selection(step)) for step in range(STEPS)]
    write(root / "sampling-plan.json", sampling)
    bound[str(root / "sampling-plan.json")] = sha(root / "sampling-plan.json")
    write(root / "train.json", dict(cases=COUNT, seed=SEED, noise_seed=NOISE_SEED, steps=STEPS))
    write(
        root / "probe.json",
        dict(cases=COUNT, seed=EVAL_SEED, modes=MODES, draws=2, depth=3, confirmation=False),
    )
    write(
        root / "plan.json",
        dict(
            initial=str(root / "initial"),
            initial_files={p.name: sha(p) for p in (root / "initial").iterdir()},
            parent=str(parent),
            data={},
            bound_files=bound,
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
    for p, h in plan["bound_files"].items():
        assert sha(Path(p)) == h, p
    for r in read(root / "data.json")["records"]:
        assert sha(root / r["file"]) == r["sha256"]
    return plan


def train(root):
    check(root)
    assert os.environ.get("XLA_FLAGS", "") == "" and jax.devices()[0].platform != "cpu"
    assert (
        read(root / "test-results.json")["passed"]
        and read(root / "metric-specification.json")["passed"]
    )
    arms = make_arms(root)
    with np.load(root / "train.npz") as z:
        data = {k: z[k] for k in z.files}
    initial = []
    for level in range(6):
        p, _ = batch(data, level)
        values = [forward(a["model"], p) for a in arms.values()]
        for v in values[1:]:
            exact(v, values[0])
        initial.append(digest(values[0]))
    write(
        root / "initial-audit.json",
        dict(
            six_depth_outputs_exact=True,
            digests=initial,
            parameters=15809,
            critic_parameters=sum(
                x.size for x in jax.tree.leaves(nnx.state(arms["prior"]["critic"], nnx.Param))
            ),
        ),
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
            rows = [json.loads(s) for s in journal.read_text().splitlines()]
            rows = [r for r in rows if r["step"] <= start]
            assert [r["step"] for r in rows] == list(range(1, start + 1))
            journal.write_text("".join(json.dumps(r) + "\n" for r in rows))
        if start == 0:
            lineart_state.save(saved, arms, 0)
            snapshot(root, arms, 0)
        prefix = []
        for step in range(start, STEPS):
            p, coords = batch(data, step)
            stats = {}
            times = {}
            order = METHODS[step % 3 :] + METHODS[: step % 3]
            for name in order:
                tick = time.perf_counter()
                stats[name] = update(arms[name], name, p, coords, step)
                times[name] = time.perf_counter() - tick
            row = dict(
                step=step + 1,
                **selection(step),
                input_sha256=digest((p, coords)),
                metrics=stats,
                seconds=times,
            )
            with journal.open("a") as f:
                f.write(json.dumps(row) + "\n")
            if step < 6:
                prefix.append(stats)
            if step < 6 or (step + 1) % 100 == 0:
                progress = dict(step=step + 1, loss={m: stats[m]["loss"] for m in METHODS})
                write(out / "progress.json", progress)
                print(progress, flush=True)
            if step + 1 == 6:
                replay = make_arms(root)
                metric_audits = []
                for k in range(6):
                    q, c = batch(data, k)
                    actual = {m: update(replay[m], m, q, c, k) for m in METHODS}
                    if start == 0:
                        metric_audits.append(
                            {m: replay_comparison(actual[m], prefix[k][m], m) for m in METHODS}
                        )
                expected_state = lineart_state.payload(arms, 6)["arms"]
                actual_state = lineart_state.payload(replay, 6)["arms"]
                state_audits = {
                    m: replay_comparison(actual_state[m], expected_state[m], m) for m in METHODS
                }
                write(
                    root / "prefix-audit.json",
                    dict(
                        passed=True,
                        metrics_checked=start == 0,
                        metrics=metric_audits,
                        states=state_audits,
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
                    actual, c = load_inference(folder)
                    assert c == arm["config"]
                    exact(nnx.state(expected, nnx.Param), nnx.state(actual, nnx.Param))
                    bindings |= {str(p): sha(p) for p in folder.iterdir() if p.is_file()}
        write(root / "snapshot-audit.json", dict(raw_ema_exact=True, exports=24, bindings=bindings))
    write(root / "workflow-status.json", dict(stage="training_complete", step=STEPS))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "freeze", "train"))
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    r = a.root.resolve()
    if a.action == "train":
        with (r / "process.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            write(r / "workflow-status.json", dict(stage="training"))
            train(r)
    elif a.action == "freeze":
        freeze(r)
    else:
        prepare(r)
