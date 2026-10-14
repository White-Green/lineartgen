"""Matched fresh plain and D4 group geometry models on sixteen known shapes."""

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
from cue_preserving_data import strength_map
from d4_geometry_data import transform_directions, transform_numpy_batch, transform_prepared
from diagnose_learnability import freeze, verify
from flax import nnx
from group_field_model import GroupFieldCNN
from PIL import Image, ImageDraw
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
STEPS = 6000
SAVED = (0, 6, 2000, STEPS)
SEED = 2026101433
INDICES = (464, 473, 482, 491, 500, 509, 466, 479, 288, 299, 310, 321, 240, 249, 432, 448)
PARENT = Path("../jax-region-sign-geometry-20261014/runs/region-sign-geometry-20261014")
REFERENCE = Path("../jax-d4-geometry-20261014/runs/d4-geometry-20261014")
ORIENTATION = Path("../jax-d4-orientation-recheck-20261014/runs/d4-orientation-recheck-20261014")
DOCUMENT = Path("docs/group-geometry-plan-20261014.md")


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
        indices=rng.integers(16, size=4).tolist(),
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
        jnp.asarray(data["clean"][ii]),
        jnp.asarray(source),
        jnp.full(source.shape, 0.5),
        jax.random.fold_in(jax.random.PRNGKey(SEED), step),
        s["depth"],
    )
    ann = jnp.asarray(data["directions"][ii])
    return transform_prepared(p, codes), candidates, transform_directions(ann, codes)


def load_data(root):
    with np.load(root / "data.npz") as z:
        return {k: z[k] for k in z.files}


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
    assert read(parent / "completion.json")["stage"] == "complete_region_sign_geometry"
    assert read(REFERENCE / "completion.json")["stage"] == "complete_d4_geometry"
    assert read(ORIENTATION / "completion.json")["stage"] == "complete_d4_orientation_recheck"
    root.mkdir(parents=True, exist_ok=False)
    (root / "configs").mkdir()
    shutil.copytree(REFERENCE / "snapshots/step-12000/d4/raw", root / "reference")
    _, cfg = load_field(root / "reference")
    train = cfg.train.model_copy(
        update=dict(
            steps=STEPS,
            learning_rate_decay_steps=STEPS,
            seed=SEED,
            model_seed=MODEL_SEED,
            learning_rate=5e-5,
            minimum_learning_rate=1e-5,
        )
    )
    cfg = cfg.model_copy(update=dict(train=train))
    assert cfg.train.peak_learning_rate == 1e-4 and cfg.model.pyramid_conditioning == "fourier"
    for method in METHODS:
        c = cfg.model_copy(
            update=dict(model=cfg.model.model_copy(update=dict(channels=(WIDTHS[method],))))
        )
        model = (
            create_field_model(c, seed=MODEL_SEED)
            if method == "plain"
            else GroupFieldCNN(c.model, rngs=nnx.Rngs(MODEL_SEED))
        )
        assert (
            sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param))) == PARAMETERS[method]
        )
        save_field(
            model,
            c,
            root / f"initials/{method}",
            provenance=f"Fresh {method} seed{MODEL_SEED}; research only",
        )
        loaded, cc = load_field(root / f"initials/{method}")
        assert cc == c
        exact(nnx.state(model, nnx.Param), nnx.state(loaded, nnx.Param))
        save_config(c, root / f"configs/{method}.json")
    for name in ("data.npz", "cases.json", "evaluation-inputs.npz"):
        shutil.copy2(parent / name, root / name)
    write(root / "sampling-plan.json", [dict(step=k, **select(k)) for k in range(STEPS)])
    write(
        root / "train.json",
        dict(
            cases=16,
            steps=STEPS,
            seed=SEED,
            model_seed=MODEL_SEED,
            training_only=True,
            parameters=PARAMETERS,
            augmentation="shared D4 + rough",
            fresh_initialization=True,
            same_initial_function=True,
            same_initial_parameters=False,
        ),
    )
    write(
        root / "probe.json",
        dict(
            cases=16,
            depths=[3, 6],
            confirmation=False,
            modes=["whole05", "partial05"],
            training_only=True,
            reference="EC d4 raw12000, unmatched historical model",
        ),
    )
    bound = {str(p): sha(p) for p in root.rglob("*") if p.is_file()}
    for folder, names in (
        (parent, ("completion.json", "source.json", "data.npz")),
        (REFERENCE.resolve(), ("completion.json", "diagnostics/review-observations.json")),
        (ORIENTATION.resolve(), ("completion.json", "evaluation/visual-review.json")),
    ):
        for name in names:
            bound[str(folder / name)] = sha(folder / name)
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
            training_only=True,
            parent=str(parent),
            seed=SEED,
        ),
    )


def check(root):
    verify(root)
    p = read(root / "plan.json")
    assert sha(DOCUMENT) == p["plan_document_sha256"]
    for f, h in p["bound_files"].items():
        assert sha(Path(f)) == h, f


def train(root):
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
        for step in range(start, STEPS):
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
        write(root / "snapshot-audit.json", dict(raw_ema_exact=True, exports=24, bindings=bindings))
    write(root / "workflow-status.json", dict(stage="training_complete", step=STEPS))


def evaluate(root):
    check(root)
    data = load_data(root)
    with np.load(root / "evaluation-inputs.npz") as z:
        sources = {k: z[k] for k in z.files}
    models = {"initial": load_field(root / "reference")[0]}
    models.update({m: load_field(root / f"snapshots/step-{STEPS}/{m}/raw")[0] for m in METHODS})
    cases = read(root / "cases.json")
    out = root / "evaluation"
    out.mkdir(exist_ok=True)
    rows, bindings = [], {}
    for condition, source in sources.items():
        for mode in ("whole05", "partial05"):
            for depth in (3, 6):
                for first in range(0, 16, 4):
                    ii = np.arange(first, first + 4)
                    s = jnp.asarray(source[ii])
                    t = jnp.asarray(np.stack([strength_map(mode)] * 4))
                    p = prepare_inputs(
                        jnp.asarray(data["clean"][ii]), s, t, jax.random.PRNGKey(SEED + 1), depth
                    )
                    arrays = {}
                    for name, model in models.items():
                        pred = np.asarray(infer(model, p))
                        assert np.isfinite(pred).all()
                        protected = np.asarray(t) == 0
                        np.testing.assert_array_equal(pred[protected], np.asarray(s)[protected])
                        arrays[name] = pred
                        for i in ii:
                            rows.append(
                                dict(
                                    condition=condition,
                                    mode=mode,
                                    depth=depth,
                                    index=int(i),
                                    kind=cases[i]["kind"],
                                    model=name,
                                )
                            )
                    path = out / f"{condition}-{mode}-d{depth}-{first}.npz"
                    np.savez_compressed(path, **arrays)
                    bindings[str(path)] = sha(path)
                print(dict(evaluation=condition, mode=mode, depth=depth), flush=True)
    records = []
    for condition, source in sources.items():
        for mode in ("whole05", "partial05"):
            for first in range(0, 16, 4):
                path = out / f"{condition}-{mode}-d6-{first}.npz"
                with np.load(path) as z:
                    pred = {k: z[k] for k in z.files}
                canvas = Image.new("RGB", (256 * 6, 34 + 4 * 280), "white")
                draw = ImageDraw.Draw(canvas)
                for col, label in enumerate(
                    (
                        "source",
                        "one permitted 2px clean",
                        "EC D4 historical reference",
                        "fresh plain 16ch",
                        "fresh D4 group 2x8ch",
                        "fresh D4 group 6x8ch",
                    )
                ):
                    draw.text((col * 256 + 6, 8), label, fill="black")
                t = strength_map(mode)
                for row, i in enumerate(range(first, first + 4)):
                    clean = np.where(t > 0, data["clean"][i], source[i])
                    arrays = [
                        source[i],
                        clean,
                        pred["initial"][row],
                        pred["plain"][row],
                        pred["group2"][row],
                        pred["group"][row],
                    ]
                    y = 34 + row * 280
                    for col, value in enumerate(arrays):
                        px = np.uint8(np.round(np.clip((value[..., 0] + 1) * 127.5, 0, 255)))
                        canvas.paste(Image.fromarray(px).convert("RGB"), (col * 256, y + 20))
                    draw.text(
                        (6, y),
                        f"{i} {cases[i]['kind']} / {condition} / {mode} / depth6",
                        fill="black",
                    )
                image_path = out / f"native-{condition}-{mode}-{first}.png"
                canvas.save(image_path)
                records.append(
                    dict(file=image_path.name, sha256=sha(image_path), prediction_sha256=sha(path))
                )
    assert len(rows) == 512 and len(records) == 16
    write(out / "manifest.json", dict(records=records))
    write(
        out / "metrics.json",
        dict(rows=rows, model_outputs=len(rows), protected_exact=True, bindings=bindings),
    )
    write(root / "workflow-status.json", dict(stage="evaluation_complete_awaiting_visual_review"))


def finalize(root):
    check(root)
    records = [json.loads(s) for s in (root / "training/sampling.jsonl").read_text().splitlines()]
    assert [r["step"] for r in records] == list(range(1, STEPS + 1))
    assert [dict(step=r["step"] - 1, **{k: r[k] for k in select(0)}) for r in records] == read(
        root / "sampling-plan.json"
    )
    for r in records:
        assert len(set(r["shared_target_sha256"].values())) == 1
        assert len(set(r["input_sha256"].values())) == 1
    assert read(root / "prefix-audit.json")["passed"]
    assert read(root / "initial-audit.json")["six_depth_functions_exact"]
    tests = read(root / "test-results.json")
    assert tests["passed"]
    for f, h in tests["logs"].items():
        assert sha(root / f) == h
    audit = read(root / "snapshot-audit.json")
    assert audit["raw_ema_exact"] and audit["exports"] == 24
    for f, h in audit["bindings"].items():
        assert sha(Path(f)) == h
    for step in SAVED[1:]:
        assert read(root / f"training/restore-{step}.json")["full_state_exact"]
    out = root / "evaluation"
    metrics = read(out / "metrics.json")
    assert metrics["model_outputs"] == 512 and metrics["protected_exact"]
    for f, h in metrics["bindings"].items():
        assert sha(Path(f)) == h
    review = read(out / "visual-review.json")
    manifest = read(out / "manifest.json")["records"]
    assert review["reviewed"] and len(review["observations"]) == len(manifest) == 16
    for r in manifest:
        assert sha(out / r["file"]) == r["sha256"] == review["files"][r["file"]]
        assert r["file"] in review["observations"]
    evidence = {
        str(p.relative_to(root)): sha(p)
        for p in (
            root / "source.json",
            root / "plan.json",
            root / "test-results.json",
            root / "initial-audit.json",
            root / "prefix-audit.json",
            root / "snapshot-audit.json",
            root / "training/sampling.jsonl",
            out / "metrics.json",
            out / "manifest.json",
            out / "visual-review.json",
        )
    }
    write(
        root / "completion.json",
        dict(
            stage="complete_group_geometry",
            training_only=True,
            steps_per_arm=STEPS,
            model_outputs=512,
            native_images_reviewed=16,
            adoption=False,
            evidence=evidence,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "freeze", "train", "evaluate", "finalize"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "freeze":
        freeze(root)
    elif args.action == "train":
        with (root / "process.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            train(root)
    else:
        globals()[args.action](root)
