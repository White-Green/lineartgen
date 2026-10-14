"""Fresh optimized architectures on the complete existing synthetic development set."""

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
from group_field_model import GroupFieldCNN
from plausible_study import digest, exact
from regional_field_objective import regional_gradient
from rough_family_data import BASE, augmentation, rough_source
from scale_style_balance_study import update
from subpixel_field_model import create_field_model, infer, load_field, save_field

from lineartgen.config import load_config, save_config
from lineartgen.training import checkpoints, lineart_state

METHODS = ("plain", "group2", "group")
WIDTHS = {"plain": 16, "group2": 2, "group": 6}
PARAMETERS = {"plain": 16064, "group2": 2022, "group": 16402}
MODEL_SEED = 2026101432
STEPS = 18000
SAVED = (0, 6, 4500, 9000, 13500, STEPS)
SEED = 2026101440
EVAL_SEED = 2026101441
PARENT = Path("../jax-d4-geometry-20261014/runs/d4-geometry-20261014")
TRANSFER = Path("../jax-optimized-transfer-20261014/runs/optimized-transfer-20261014")
DOCUMENT = Path("docs/optimized-broad-plan-20261014.md")


def initialize(config, method):
    model = (
        create_field_model(config, seed=MODEL_SEED)
        if method == "plain"
        else GroupFieldCNN(config.model, rngs=nnx.Rngs(MODEL_SEED))
    )
    head = model.head if method == "plain" else model.head.linear
    head.bias[...] = jnp.full_like(head.bias[...], 0.125)
    return model


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
    return dict(
        indices=rng.integers(512, size=4).tolist(),
        modes=rng.integers(len(MODES), size=4).tolist(),
        depth=step % 6 + 1,
        augmentation=[augmentation(SEED, step, i) for i in range(4)],
        transforms=rng.integers(8, size=4).tolist(),
    )


def batch(data, step, method):
    assert method in METHODS
    s = select(step)
    ii = np.asarray(s["indices"])
    source = np.stack(
        [
            data["source"][i]
            if opts == BASE
            else rough_source(data["variants"][i, [1, 25]].mean(0), opts)
            for i, opts in zip(ii, s["augmentation"], strict=True)
        ]
    )
    codes = jnp.asarray(s["transforms"])
    candidates = jnp.asarray(
        transform_numpy_batch(data["fields"][ii, ..., None], s["transforms"])[..., 0]
    )
    p = prepare_inputs(
        jnp.asarray(data["variants"][ii, 13]),
        jnp.asarray(source),
        jnp.asarray(np.stack([strength_map(MODES[m]) for m in s["modes"]])),
        jax.random.fold_in(jax.random.PRNGKey(SEED), step),
        s["depth"],
    )
    ann = jnp.asarray(data["directions"][ii])
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
    transfer = TRANSFER.resolve()
    assert read(parent / "completion.json")["stage"] == "complete_d4_geometry"
    assert read(transfer / "completion.json")["stage"] == "complete_optimized_transfer"
    root.mkdir(parents=True, exist_ok=False)
    (root / "configs").mkdir()
    shutil.copytree(parent / "snapshots/step-12000/d4/raw", root / "reference")
    _, cfg = load_field(root / "reference")
    train = cfg.train.model_copy(
        update=dict(
            steps=STEPS,
            learning_rate_decay_steps=STEPS,
            seed=SEED,
            model_seed=MODEL_SEED,
            learning_rate=5e-4,
            minimum_learning_rate=1e-4,
        )
    )
    cfg = cfg.model_copy(update=dict(train=train))
    assert cfg.train.peak_learning_rate == 1e-3 and cfg.model.pyramid_conditioning == "fourier"
    for method in METHODS:
        c = cfg.model_copy(
            update=dict(model=cfg.model.model_copy(update=dict(channels=(WIDTHS[method],))))
        )
        model = initialize(c, method)
        assert (
            sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param))) == PARAMETERS[method]
        )
        save_field(
            model,
            c,
            root / f"initials/{method}",
            provenance=f"Fresh {method} seed{MODEL_SEED}; broad optimized research only",
        )
        loaded, cc = load_field(root / f"initials/{method}")
        assert cc == c
        exact(nnx.state(model, nnx.Param), nnx.state(loaded, nnx.Param))
        save_config(c, root / f"configs/{method}.json")
    # Immutable existing data are reused by hash; no additional large field-cache copy.
    for name in (
        "train.npz",
        "data.json",
        "stress-plan.json",
        "fields.npy",
        "directions.npy",
        "cache-audit.json",
    ):
        (root / name).symlink_to(parent / name)
    bound = {}
    for name in ("data", "stress-data"):
        (root / name).symlink_to(parent / name, target_is_directory=True)
        for p in (parent / name).rglob("*"):
            if p.is_file():
                bound[str(p)] = sha(p)
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
            model_seed=MODEL_SEED,
            parameters=PARAMETERS,
            augmentation="shared D4 + 25 percent base / 75 percent rough",
            modes=MODES,
            fresh_initialization=True,
            same_initial_function=True,
            same_initial_parameters=False,
            objective="unchanged regional censored geometry + RMS near/far sign",
            reference="EC d4 raw12000, unmatched historical continued-training model",
        ),
    )
    write(
        root / "probe.json",
        dict(cases=256, seed=EVAL_SEED, modes=MODES, draws=2, depth=3, confirmation=False),
    )
    bound.update({str(p): sha(p) for p in root.rglob("*") if p.is_file()})
    for p in (
        parent / "completion.json",
        parent / "source.json",
        transfer / "completion.json",
        transfer / "evaluation/visual-review.json",
    ):
        bound[str(p)] = sha(p)
    initial = root / "initials/plain"
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
            outputs=12544,
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
    assert until in (4500, 9000, 13500, STEPS)
    check(root)
    assert os.environ.get("XLA_FLAGS", "") == "" and jax.devices()[0].platform != "cpu"
    assert read(root / "test-results.json")["passed"]
    data, arms = load_data(root), arms_from_root(root)
    initial_predictions = {}
    for k in range(6):
        prepared, _, _ = batch(data, k, "plain")
        a = np.asarray(infer(arms["plain"]["model"], prepared))
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
            assert len(set(shared.values())) == 1
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
        write(root / "snapshot-audit.json", dict(raw_ema_exact=True, exports=36, bindings=bindings))
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
