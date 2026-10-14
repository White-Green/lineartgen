"""Training-only focused fit: fixed versus varying rough source, same 16 geometries."""

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
from cue_capacity import forward, make_arm, prepare_inputs
from cue_preserving_data import strength_map
from diagnose_learnability import freeze, snapshot, verify
from flax import nnx
from PIL import Image, ImageDraw
from plausible_study import digest, exact, replay_comparison
from rough_family_data import BASE, augmentation, rough_source, stress_options
from scale_style_balance_study import update
from style_family_objective import style_gradient, style_path_costs, style_pixel_costs

from lineartgen.config import load_config, save_config
from lineartgen.models.serialization import load_inference
from lineartgen.training import checkpoints, lineart_state

METHODS = ("fixed", "augmented")
STEPS = 3000
SAVED = (0, 6, 1000, STEPS)
SEED = 2026101404
INDICES = (464, 473, 482, 491, 500, 509, 466, 479, 288, 299, 310, 321, 240, 249, 432, 448)
PARENT = Path("../jax-scale-style-balance-20261014/runs/scale-style-balance-20261014")
DOCUMENT = Path("docs/focused-fit-plan-20261014.md")


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
    )


def batch(data, step, method):
    assert method in METHODS
    s = select(step)
    ii = np.asarray(s["indices"])
    source = np.stack(
        [
            data["source"][i]
            if method == "fixed" or opts == BASE
            else rough_source(data["variants"][i, [1, 25]].mean(0), opts)
            for i, opts in zip(ii, s["augmentation"], strict=True)
        ]
    )
    candidates = jnp.asarray(data["allstroke"][ii])
    p = prepare_inputs(
        candidates[:, 13],
        jnp.asarray(source),
        jnp.full(source.shape, 0.5),
        jax.random.fold_in(jax.random.PRNGKey(SEED), step),
        s["depth"],
    )
    ann = {k: jnp.asarray(data[k][ii]) for k in ("probes", "valid")}
    ann["normalizers"] = candidates
    return p, candidates, ann


def load_data(root):
    with np.load(root / "data.npz") as z:
        return {k: z[k] for k in z.files}


def arms_from_root(root):
    model, _ = load_inference(root / "initial")
    arms = {}
    for method in METHODS:
        arms[method] = make_arm(load_config(root / f"configs/{method}.json"), model)
        arms[method]["gradient"] = style_gradient()
    return arms


def prepare(root):
    parent = PARENT.resolve()
    assert read(parent / "completion.json")["stage"] == "complete_scale_style_balance"
    root.mkdir(parents=True, exist_ok=False)
    (root / "configs").mkdir()
    shutil.copytree(parent / "snapshots/step-18000/both/raw", root / "initial")
    model, cfg = load_inference(root / "initial")
    train = cfg.train.model_copy(
        update=dict(steps=STEPS, learning_rate_decay_steps=STEPS, seed=SEED)
    )
    cfg = cfg.model_copy(update=dict(train=train))
    assert cfg.train.peak_learning_rate == 1e-4 and cfg.model.pyramid_conditioning == "fourier"
    for method in METHODS:
        save_config(cfg, root / f"configs/{method}.json")
    data = {}
    for name, keys in (
        ("train.npz", ("source", "variants")),
        ("allstroke.npz", ("allstroke",)),
        ("path-annotations.npz", ("probes", "valid")),
    ):
        with np.load(parent / name) as z:
            data.update({k: z[k][list(INDICES)] for k in keys})
    np.savez_compressed(root / "data.npz", **data)
    rows = [r for r in read(parent / "data.json")["records"] if r["split"] == "expanded"]
    assert [rows[i]["kind"] for i in INDICES] == ["wide"] * 8 + ["parallel"] * 4 + [
        "crossing"
    ] * 2 + ["blank", "black"]
    write(
        root / "cases.json",
        [dict(index=i, parent_index=j, kind=rows[j]["kind"]) for i, j in enumerate(INDICES)],
    )
    ev = dict(
        source=data["source"],
        rough=np.stack(
            [
                rough_source(v[[1, 25]].mean(0), stress_options("rough", i))
                for i, v in enumerate(data["variants"])
            ]
        ),
    )
    np.savez_compressed(root / "evaluation-inputs.npz", **ev)
    write(root / "sampling-plan.json", [dict(step=k, **select(k)) for k in range(STEPS)])
    write(root / "train.json", dict(cases=16, steps=STEPS, seed=SEED, training_only=True))
    write(
        root / "probe.json",
        dict(
            cases=16,
            depths=[3, 6],
            confirmation=False,
            modes=["whole05", "partial05"],
            training_only=True,
        ),
    )
    bound = {str(p): sha(p) for p in root.rglob("*") if p.is_file()}
    for name in (
        "completion.json",
        "source.json",
        "snapshot-audit.json",
        "data.json",
        "train.npz",
        "allstroke.npz",
        "path-annotations.npz",
        "diagnostics/review-observations.json",
    ):
        bound[str(parent / name)] = sha(parent / name)
    for p in (parent / "snapshots/step-18000/both/raw").iterdir():
        bound[str(p)] = sha(p)
    write(
        root / "plan.json",
        dict(
            initial=str(root / "initial"),
            initial_files={p.name: sha(p) for p in (root / "initial").iterdir()},
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
    exact(
        nnx.state(arms["fixed"]["model"], nnx.Param),
        nnx.state(arms["augmented"]["model"], nnx.Param),
    )
    write(
        root / "backend.json",
        dict(
            device=jax.devices()[0].device_kind,
            parameters=15809,
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
                    audits.append(
                        {m: replay_comparison(actual[m], prefix[k][m], "prior") for m in METHODS}
                    )
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
                    actual, cfg = load_inference(folder)
                    assert cfg == arm["config"]
                    exact(expected, nnx.state(actual, nnx.Param))
                    bindings.update({str(p): sha(p) for p in folder.iterdir() if p.is_file()})
        write(root / "snapshot-audit.json", dict(raw_ema_exact=True, exports=16, bindings=bindings))
    write(root / "workflow-status.json", dict(stage="training_complete", step=STEPS))


@nnx.jit
def measure(model, p, candidates, ann):
    pred, raw = forward(model, p)
    pixels = style_pixel_costs(raw, candidates, p.noise_levels, ann["normalizers"])
    profiles = style_path_costs(pred, candidates, p.noise_levels, ann, ann["normalizers"])
    costs = pixels + 0.25 * profiles
    return pred[0], costs.min(1), costs.argmin(1)


def evaluate(root):
    check(root)
    data = load_data(root)
    with np.load(root / "evaluation-inputs.npz") as z:
        sources = {k: z[k] for k in z.files}
    models = {"initial": load_inference(root / "initial")[0]}
    models.update({m: load_inference(root / f"snapshots/step-{STEPS}/{m}/raw")[0] for m in METHODS})
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
                    c = jnp.asarray(data["allstroke"][ii])
                    ann = {k: jnp.asarray(data[k][ii]) for k in ("probes", "valid")}
                    ann["normalizers"] = c
                    p = prepare_inputs(c[:, 13], s, t, jax.random.PRNGKey(SEED + 1), depth)
                    arrays = {}
                    for name, model in models.items():
                        pred, cost, choice = jax.device_get(measure(model, p, c, ann))
                        assert np.isfinite(pred).all() and np.isfinite(cost).all()
                        protected = np.asarray(t) == 0
                        np.testing.assert_array_equal(pred[protected], np.asarray(s)[protected])
                        arrays[name] = pred
                        for j, i in enumerate(ii):
                            rows.append(
                                dict(
                                    condition=condition,
                                    mode=mode,
                                    depth=depth,
                                    index=int(i),
                                    kind=cases[i]["kind"],
                                    model=name,
                                    cost=float(cost[j]),
                                    choice=int(choice[j]),
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
                canvas = Image.new("RGB", (256 * 5, 34 + 4 * 280), "white")
                draw = ImageDraw.Draw(canvas)
                for col, label in enumerate(
                    (
                        "source",
                        "one permitted 2px clean",
                        "DN both initial",
                        "fixed fit",
                        "augmented fit",
                    )
                ):
                    draw.text((col * 256 + 6, 8), label, fill="black")
                t = strength_map(mode)
                for row, i in enumerate(range(first, first + 4)):
                    clean = np.where(t > 0, data["allstroke"][i, 13], source[i])
                    arrays = [
                        source[i],
                        clean,
                        pred["initial"][row],
                        pred["fixed"][row],
                        pred["augmented"][row],
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
    assert len(rows) == 384 and len(records) == 16
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
    assert read(root / "prefix-audit.json")["passed"]
    audit = read(root / "snapshot-audit.json")
    assert audit["raw_ema_exact"] and audit["exports"] == 16
    for f, h in audit["bindings"].items():
        assert sha(Path(f)) == h
    for step in SAVED[1:]:
        assert read(root / f"training/restore-{step}.json")["full_state_exact"]
    out = root / "evaluation"
    metrics = read(out / "metrics.json")
    assert metrics["model_outputs"] == 384 and metrics["protected_exact"]
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
            stage="complete_focused_fit",
            training_only=True,
            steps_per_arm=STEPS,
            model_outputs=384,
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
