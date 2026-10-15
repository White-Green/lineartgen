"""Matched continuation with or without union composition of existing training geometry."""

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
from campaign_protocol import read, sha, write
from cue_capacity import make_arm, prepare_inputs
from cue_preserving_data import MODES, strength_map
from d4_geometry_data import transform_directions, transform_numpy_batch, transform_prepared
from diagnose_learnability import freeze, verify
from flax import nnx
from plausible_study import digest, exact
from regional_field_objective import regional_gradient
from rough_family_data import BASE, augmentation, rough_source
from scale_style_balance_study import update
from subpixel_field_model import infer, load_field, save_field

from lineartgen.config import load_config, save_config
from lineartgen.training import checkpoints, lineart_state

METHODS = ("control", "composed")
PARAMETERS = {m: 16402 for m in METHODS}
STEPS = 6000
SAVED = (0, 6, 3000, STEPS)
SEED = 2026101502
EVAL_SEED = 2026101503
PARENT = Path("../jax-optimized-broad-20261014/runs/optimized-broad-20261014")
DOCUMENT = Path("docs/geometry-coverage-plan-20261015.md")


def replay_state(actual, expected):
    """Warm ROCm replay envelope established before this fresh diagnostic run."""
    aa, sa = jax.tree.flatten(actual)
    bb, sb = jax.tree.flatten(expected)
    assert sa == sb
    maximum = 0.0
    for a, b in zip(aa, bb, strict=True):
        a, b = np.asarray(a), np.asarray(b)
        assert a.dtype == b.dtype and a.shape == b.shape
        if np.issubdtype(a.dtype, np.inexact):
            np.testing.assert_allclose(a, b, atol=5e-6, rtol=1e-4, equal_nan=False)
            maximum = max(maximum, float(np.max(abs(a - b), initial=0)))
        else:
            np.testing.assert_array_equal(a, b)
    return dict(
        passed=True,
        max_abs_difference=maximum,
        atol=5e-6,
        rtol=1e-4,
        exact=all(np.array_equal(a, b) for a, b in zip(aa, bb, strict=True)),
    )


def select(step):
    rng = np.random.default_rng([SEED, step])
    indices = rng.integers(512, size=4)
    modes = rng.integers(len(MODES), size=4)
    transforms = rng.integers(8, size=4)
    companions = rng.integers(432, size=4)
    # Fills, blanks and wide guards keep their original task distribution.
    compose = (rng.random(4) < 0.5) & (indices < 432)
    return dict(
        indices=indices.tolist(),
        modes=modes.tolist(),
        depth=step % 6 + 1,
        augmentation=[augmentation(SEED, step, i) for i in range(4)],
        transforms=transforms.tolist(),
        companions=companions.tolist(),
        compose=compose.tolist(),
    )


def components(data, selection, method):
    assert method in METHODS
    sources, clean, fields = [], [], []
    for i, j, compose, opts in zip(
        selection["indices"],
        selection["companions"],
        selection["compose"],
        selection["augmentation"],
        strict=True,
    ):
        indices = (i, j) if method == "composed" and compose else (i,)
        inputs = [
            data["source"][k]
            if opts == BASE
            else rough_source(data["variants"][k, [1, 25]].mean(0), opts)
            for k in indices
        ]
        sources.append(np.minimum.reduce(inputs))
        clean.append(np.minimum.reduce([data["variants"][k, 13] for k in indices]))
        fields.append(np.minimum.reduce([data["fields"][k] for k in indices]))
    return np.stack(sources), np.stack(clean), np.stack(fields)


def batch(data, step, method):
    s = select(step)
    source, clean, fields = components(data, s, method)
    codes = jnp.asarray(s["transforms"])
    candidates = jnp.asarray(transform_numpy_batch(fields[..., None], s["transforms"])[..., 0])
    p = prepare_inputs(
        jnp.asarray(clean),
        jnp.asarray(source),
        jnp.asarray(np.stack([strength_map(MODES[m]) for m in s["modes"]])),
        jax.random.fold_in(jax.random.PRNGKey(SEED), step),
        s["depth"],
    )
    ann = jnp.asarray(data["directions"][s["indices"]])
    return transform_prepared(p, codes), candidates, transform_directions(ann, codes)


def load_data(root):
    with np.load(root / "train.npz") as z:
        data = {k: z[k] for k in z.files}
    data["fields"] = np.load(root / "fields.npy", mmap_mode="r")
    data["directions"] = np.load(root / "directions.npy")
    return data


def arms_from_root(root):
    arms = {}
    for method in METHODS:
        model, cfg = load_field(root / f"initials/{method}")
        assert cfg == load_config(root / f"configs/{method}.json")
        arms[method] = make_arm(cfg, model)
        arms[method]["gradient"] = regional_gradient()
    return arms


def snapshot(root, arms, step):
    for name, arm in arms.items():
        for mode in ("raw", "ema"):
            model = nnx.clone(arm["model"])
            if mode == "ema":
                nnx.update(model, arm["ema"])
            save_field(
                model,
                arm["config"],
                root / f"snapshots/step-{step}/{name}/{mode}",
                provenance=f"regional geometry {name} step{step} {mode}; training-only diagnostic",
            )


def prepare(root):
    parent = PARENT.resolve()
    assert read(parent / "completion.json")["stage"] == "complete_optimized_broad"
    root.mkdir(parents=True, exist_ok=False)
    (root / "configs").mkdir()
    shutil.copytree(parent / "snapshots/step-18000/group/raw", root / "reference")
    model, cfg = load_field(root / "reference")
    train = cfg.train.model_copy(
        update=dict(
            steps=STEPS,
            learning_rate_decay_steps=STEPS,
            seed=SEED,
            learning_rate=1e-4,
            minimum_learning_rate=2e-5,
            warmup_steps=0,
        )
    )
    cfg = cfg.model_copy(update=dict(train=train))
    assert cfg.train.peak_learning_rate == 2e-4
    for method in METHODS:
        save_field(
            model,
            cfg,
            root / f"initials/{method}",
            provenance="EI group6 raw18000; matched optimizer reset; composition study",
        )
        loaded, cc = load_field(root / f"initials/{method}")
        assert cc == cfg
        exact(nnx.state(model, nnx.Param), nnx.state(loaded, nnx.Param))
        save_config(cfg, root / f"configs/{method}.json")
    for name in (
        "train.npz",
        "data.json",
        "stress-plan.json",
        "fields.npy",
        "directions.npy",
        "cache-audit.json",
    ):
        (root / name).symlink_to((parent / name).resolve())
    bound = {}
    for name in ("data", "stress-data"):
        (root / name).symlink_to((parent / name).resolve(), target_is_directory=True)
        for p in (parent / name).rglob("*"):
            if p.is_file():
                bound[str(p.resolve())] = sha(p)
    (root / "scene512").mkdir()
    for name in ["plan.json"] + [f"input-{i}.npz" for i in range(8)]:
        shutil.copy2(parent / "scene512" / name, root / "scene512" / name)
    write(root / "sampling-plan.json", [dict(step=k, **select(k)) for k in range(STEPS)])
    write(
        root / "train.json",
        dict(
            cases=512,
            steps=STEPS,
            seed=SEED,
            parameters=PARAMETERS,
            augmentation="D4 + 25/75 base/rough; .5 union probability for ordinary shapes",
            modes=MODES,
            fresh_initialization=False,
            same_initial_parameters=True,
            objective="unchanged regional censored geometry + RMS near/far sign",
            reference="EI group6 raw18000",
            optimizer_reset=True,
        ),
    )
    write(
        root / "probe.json",
        dict(cases=256, seed=EVAL_SEED, modes=MODES, draws=2, depth=3, confirmation=False),
    )
    bound.update({str(p): sha(p) for p in root.rglob("*") if p.is_file()})
    for p in (parent / "completion.json", parent / "source.json", DOCUMENT):
        bound[str(p.resolve())] = sha(p)
    initial = root / "initials/control"
    write(
        root / "plan.json",
        dict(
            initial=str(initial),
            initial_files={p.name: sha(p) for p in initial.iterdir()},
            data={},
            bound_files=bound,
            methods=METHODS,
            steps=STEPS,
            saved=SAVED,
            plan_document_sha256=sha(DOCUMENT),
            adoption=False,
            parent=str(parent),
            seed=SEED,
            outputs=9408,
            native_pages=28,
            fresh_confirmation=False,
        ),
    )


def check(root):
    verify(root)
    p = read(root / "plan.json")
    assert sha(DOCUMENT) == p["plan_document_sha256"]
    for f, h in p["bound_files"].items():
        assert sha(Path(f)) == h, f


def train(root, until=STEPS):
    assert until in (3000, STEPS)
    check(root)
    assert os.environ.get("XLA_FLAGS", "") == "" and jax.devices()[0].platform != "cpu"
    assert read(root / "test-results.json")["passed"]
    data, arms = load_data(root), arms_from_root(root)
    initial_predictions = {}
    for k in range(6):
        prepared, _, _ = batch(data, k, "control")
        a = np.asarray(infer(arms["control"]["model"], prepared))
        for method in METHODS[1:]:
            b = np.asarray(infer(arms[method]["model"], prepared))
            np.testing.assert_array_equal(a, b)
        initial_predictions[str(k + 1)] = digest(a)
    write(
        root / "initial-audit.json",
        dict(six_depth_functions_exact=True, prediction_sha256=initial_predictions),
    )
    for method, arm in arms.items():
        assert (
            sum(x.size for x in jax.tree.leaves(nnx.state(arm["model"], nnx.Param)))
            == PARAMETERS[method]
        )
    write(
        root / "backend.json",
        dict(
            device=jax.devices()[0].device_kind,
            parameters=PARAMETERS,
            flags=os.environ.get("XLA_FLAGS", ""),
        ),
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
        assert start < until <= STEPS
        for step in range(start, until):
            stats, inputs, elapsed, shared = {}, {}, {}, {}
            for method in METHODS[:: 1 if step % 2 == 0 else -1]:
                p, c, a = batch(data, step, method)
                inputs[method] = digest((p, c, a))
                shared[method] = digest((c, a, p.noise_levels))
                tick = time.perf_counter()
                stats[method] = update(arms[method], p, c, a)
                elapsed[method] = time.perf_counter() - tick
            # Changed source/geometry is the treatment; unchanged mask and RNG keys are shared.
            if not any(select(step)["compose"]):
                assert len(set(inputs.values())) == 1
            row = dict(
                step=step + 1,
                **select(step),
                metrics=stats,
                input_sha256=inputs,
                shared_target_sha256=shared,
                seconds=elapsed,
            )
            with journal.open("a") as f:
                f.write(json.dumps(row) + "\n")
            if step < 6:
                prefix.append(stats)
            if step < 6 or (step + 1) % 100 == 0:
                write(out / "progress.json", dict(step=step + 1, metrics=stats))
                print(dict(step=step + 1), flush=True)
            if step == 5:
                replay = arms_from_root(root)
                audits = []
                for k in range(6):
                    actual = {m: update(replay[m], *batch(data, k, m)) for m in METHODS}
                    audits.append({m: replay_state(actual[m], prefix[k][m]) for m in METHODS})
                expected, actual = (lineart_state.payload(x, 6)["arms"] for x in (arms, replay))
                write(
                    root / "prefix-audit.json",
                    dict(
                        passed=True,
                        metrics=audits,
                        states={m: replay_state(actual[m], expected[m]) for m in METHODS},
                    ),
                )
            if step + 1 in SAVED:
                lineart_state.save(saved, arms, step + 1)
                restored = arms_from_root(root)
                assert lineart_state.restore(saved, restored, step + 1) == step + 1
                exact(
                    lineart_state.payload(arms, step + 1), lineart_state.payload(restored, step + 1)
                )
                snapshot(root, arms, step + 1)
                write(out / f"restore-{step + 1}.json", dict(full_state_exact=True))
        if until < STEPS:
            write(
                root / "workflow-status.json", dict(stage="training_segment_complete", step=until)
            )
            return
        bindings = {}
        for step in SAVED:
            restored = arms_from_root(root)
            assert lineart_state.restore(saved, restored, step) == step
            for method, arm in restored.items():
                for mode in ("raw", "ema"):
                    expected = nnx.state(arm["model"], nnx.Param) if mode == "raw" else arm["ema"]
                    folder = root / f"snapshots/step-{step}/{method}/{mode}"
                    actual, cfg = load_field(folder)
                    assert cfg == arm["config"]
                    exact(expected, nnx.state(actual, nnx.Param))
                    bindings.update({str(p): sha(p) for p in folder.iterdir() if p.is_file()})
        write(root / "snapshot-audit.json", dict(raw_ema_exact=True, exports=16, bindings=bindings))
    write(root / "workflow-status.json", dict(stage="training_complete", step=STEPS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "freeze", "train"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--until", type=int, default=STEPS)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "freeze":
        freeze(root)
    elif args.action == "train":
        with (root / "process.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            train(root, until=args.until)
    else:
        globals()[args.action](root)
