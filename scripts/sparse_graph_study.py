"""HS: matched sparse-classification comparison, with explicit teacher-oracle diagnostics."""

import argparse
import datetime
import json
import shutil
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from campaign_protocol import read, sha, write
from cue_capacity import make_arm, prepare_inputs
from flax import nnx
from learned_centerline_model import (
    decode,
    infer_heads,
    load_weights,
    save_weights,
)
from learned_centerline_model import (
    gradient as control_gradient,
)
from learned_centerline_study import (
    PARTIAL,
    SAVED,
    SEED,
    STEPS,
    TRAIN,
    batch,
    crop,
    graph_metrics,
    halo,
    load_data,
    logits_from_target,
    make_model,
    strength,
)
from PIL import Image, ImageDraw
from plausible_study import exact
from sparse_graph_objective import gradient as sparse_gradient

from lineartgen.training import checkpoints, lineart_state

HR = Path("../jax-learned-centerline-20261018/runs/learned-centerline-20261018-retry2").resolve()
DOC = Path("docs/sparse-graph-loss-plan-20261018.md")
CODE = tuple(
    map(
        Path,
        [
            "scripts/sparse_graph_study.py",
            "scripts/sparse_graph_objective.py",
            "tests/test_sparse_graph_objective.py",
        ],
    )
)
METHODS = ("control", "sparse")
ORACLES = (
    "probe-12",
    "probe-24",
    "probe-36",
    "probe-37",
    "probe-59",
    "scene-1",
    "scene-2",
    "scene-7",
)


def now():
    return datetime.datetime.now(datetime.UTC).isoformat()


def prepare(root):
    assert (
        not root.exists()
        and not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    )
    previous = read(HR / "completion.json")
    assert previous["original_strict_protocol_passed"] is False
    for f, h in previous["evidence"].items():
        assert sha(HR / f) == h
    root.mkdir(parents=True)
    for name in ["config.json", "target-audit.json", "target-topology-summary.json"]:
        shutil.copy2(HR / name, root / name)
    shutil.copy2(HR / "snapshots/step-1200/raw.npz", root / "initial.npz")
    (root / "inputs").symlink_to(HR / "inputs", target_is_directory=True)
    model, cfg = make_model(root)
    assert cfg.train.peak_learning_rate == 2e-4 and cfg.train.steps == STEPS == 1200
    assert sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param))) == 22725
    records = read(HR / "plan.json")["records"]
    files = (
        [
            HR / "completion.json",
            HR / "supplemental-binding.json",
            HR / "plan.json",
            HR / "training/sampling.jsonl",
            root / "config.json",
            root / "initial.npz",
            root / "target-audit.json",
        ]
        + [root / r["file"] for r in records]
        + [HR / "outputs" / f"{r['name']}.npz" for r in records]
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(
        ["git", "archive", "--format=tar.gz", "-o", str(root / "source.tar.gz"), commit], check=True
    )
    write(
        root / "plan.json",
        dict(
            commit=commit,
            created_at=now(),
            records=records,
            bindings={str(p.resolve()): sha(p) for p in files},
            document_sha256=sha(DOC),
            code={str(p): sha(p) for p in CODE},
            archive_sha256=sha(root / "source.tar.gz"),
            parameters=22725,
            methods=METHODS,
            steps=STEPS,
            native_pages=39,
            comparison_tolerance=dict(atol=1e-5, rtol=1e-5),
            fresh_used=False,
            adoption=False,
        ),
    )


def check(root):
    plan = read(root / "plan.json")
    for f, h in plan["bindings"].items():
        assert sha(Path(f)) == h, f
    for f, h in plan["code"].items():
        assert sha(Path(f)) == h
    assert sha(DOC) == plan["document_sha256"]
    assert sha(root / "source.tar.gz") == plan["archive_sha256"]
    return plan


def compare(a, b):
    aa, bb = jax.tree.leaves(a), jax.tree.leaves(b)
    assert len(aa) == len(bb)
    maximum = 0.0
    same = True
    for x, y in zip(aa, bb, strict=True):
        x, y = np.asarray(x), np.asarray(y)
        assert x.shape == y.shape and x.dtype == y.dtype
        same = bool(same and np.array_equal(x, y))
        if x.size:
            maximum = max(maximum, float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max()))
        if x.dtype.kind in "fc":
            np.testing.assert_allclose(x, y, atol=1e-5, rtol=1e-5)
        else:
            np.testing.assert_array_equal(x, y)
    return dict(bit_exact=same, max_absolute_difference=maximum, tolerance_passed=True)


def arms(root):
    m, c = make_model(root)
    result = {}
    for name in METHODS:
        a = make_arm(c, m)
        a["gradient"] = control_gradient if name == "control" else sparse_gradient
        result[name] = a
    return result


def update(a, p, t):
    parts, g = a["gradient"](a["model"], p, t)
    parts["gradient_norm"] = optax.global_norm(g)
    values = {k: float(v) for k, v in parts.items()}
    assert all(np.isfinite(v) for v in values.values())
    _, a["ema"] = a["update"](a["model"], a["optimizer"], g, a["ema"])
    jax.block_until_ready(a["ema"])
    return values


def matched_batch(data, step, previous):
    p, t, audit = batch(data, step)
    assert audit == {k: v for k, v in previous[step].items() if k != "step"}
    return p, t, audit


def preflight(root):
    plan = check(root)
    data = load_data(root, plan)
    a, b = arms(root), arms(root)
    previous = [json.loads(s) for s in (HR / "training/sampling.jsonl").read_text().splitlines()]
    rows = []
    for step in range(6):
        p, t, audit = matched_batch(data, step, previous)
        metrics = {}
        comparisons = {}
        for name in METHODS:
            x, y = update(a[name], p, t), update(b[name], p, t)
            comparisons[name] = dict(
                metrics=compare(x, y),
                state=compare(
                    lineart_state.payload({name: a[name]}, step + 1),
                    lineart_state.payload({name: b[name]}, step + 1),
                ),
            )
            metrics[name] = x
        rows.append(dict(step=step, metrics=metrics, comparisons=comparisons, **audit))
    write(root / "preflight.json", dict(steps=6, rows=rows, all_comparisons_passed=True))


def snapshot(root, models, step, manager):
    lineart_state.save(manager, models, step)
    for name, a in models.items():
        for mode in ("raw", "ema"):
            m = nnx.clone(a["model"])
            if mode == "ema":
                nnx.update(m, a["ema"])
            path = root / "snapshots" / f"step-{step}" / name / f"{mode}.npz"
            save_weights(m, path)
            restored, _ = make_model(root)
            load_weights(restored, path)
            exact(nnx.state(m, nnx.Param), nnx.state(restored, nnx.Param))
    restored = arms(root)
    lineart_state.restore(manager, restored, step)
    exact(lineart_state.payload(models, step), lineart_state.payload(restored, step))


def train(root):
    plan = check(root)
    data = load_data(root, plan)
    models = arms(root)
    previous = [json.loads(s) for s in (HR / "training/sampling.jsonl").read_text().splitlines()]
    pre = read(root / "preflight.json")
    assert pre["all_comparisons_passed"]
    d = root / "training"
    d.mkdir(exist_ok=False)
    prefix = []
    rows = []
    with (
        checkpoints.manager(root / "checkpoints", keep=len(SAVED)) as manager,
        (d / "metrics.jsonl").open("w") as log,
        (d / "sampling.jsonl").open("w") as sample,
    ):
        snapshot(root, models, 0, manager)
        for step in range(STEPS):
            p, t, audit = matched_batch(data, step, previous)
            values = {}
            for name in METHODS:
                before = time.monotonic()
                m = update(models[name], p, t)
                values[name] = dict(**m, seconds=time.monotonic() - before)
                if step < 6:
                    prefix.append(
                        dict(
                            step=step,
                            name=name,
                            comparison=compare(m, pre["rows"][step]["metrics"][name]),
                            differences={
                                k: m[k] - pre["rows"][step]["metrics"][name][k] for k in m
                            },
                        )
                    )
            row = dict(step=step + 1, depth=step % 6 + 1, methods=values)
            rows.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            sample.write(json.dumps(dict(step=step, **audit)) + "\n")
            sample.flush()
            if step + 1 in SAVED:
                snapshot(root, models, step + 1, manager)
        write(root / "prefix-comparison.json", prefix)
    write(
        root / "training-summary.json",
        dict(
            steps=STEPS,
            depth_counts={str(i): sum(r["depth"] == i for r in rows) for i in range(1, 7)},
            saved=SAVED,
            all_input_hashes_match_hr=True,
            all_saved_states_exact=True,
            all_snapshots_exact=True,
            prefix_tolerance_passed=True,
            final=rows[-1],
            methods={
                name: dict(
                    mean_seconds_last600=float(
                        np.mean([r["methods"][name]["seconds"] for r in rows[-600:]])
                    ),
                    clip_count=sum(r["methods"][name]["gradient_norm"] > 1 for r in rows),
                    max_norm=max(r["methods"][name]["gradient_norm"] for r in rows),
                )
                for name in METHODS
            },
        ),
    )


def oracle_diagnostic(root):
    plan = check(root)
    data = load_data(root, plan)
    out = root / "oracle"
    out.mkdir(exist_ok=False)
    rows = []
    manifest = []
    for name in ORACLES:
        d = data[name]
        truth = logits_from_target(d["target0"])
        with np.load(HR / "outputs" / f"{name}.npz") as z:
            h = z["head-d3"]
            initial = z["graph-d3"]
        arrays = {}
        stats = {}
        for mode, channels in [
            ("initial", []),
            ("existence", [0, 8]),
            ("topology", [0, 4, 5, 6, 7, 8]),
            ("oracle", list(range(9))),
        ]:
            hh = h.copy()
            hh[..., channels] = truth[..., channels]
            image, st = decode(hh)
            arrays[mode] = crop(image)
            stats[mode] = st
        np.testing.assert_array_equal(arrays["initial"], initial)
        np.testing.assert_array_equal(arrays["oracle"], d["oracle"])
        p = out / f"{name}.npz"
        np.savez_compressed(p, **arrays)
        n = d["source"].shape[0]
        canvas = Image.new("RGB", (n * 4, n + 28), "#ddd")
        draw = ImageDraw.Draw(canvas)
        for col, mode in enumerate(arrays):
            draw.text((col * n + 3, 4), name + " " + mode, fill="black")
            canvas.paste(to_image(arrays[mode]), (col * n, 28))
        pic = out / f"{name}.png"
        canvas.save(pic)
        manifest.append(dict(file=pic.name, sha256=sha(pic), name=name, native_pixels=True))
        rows.append(
            dict(
                name=name,
                stats=stats,
                file=p.name,
                sha256=sha(p),
                initial_exact=True,
                oracle_exact=True,
            )
        )
    write(out / "manifest.json", manifest)
    write(out / "metrics.json", rows)


def to_image(a):
    return Image.fromarray(
        np.clip(np.rint((a[..., 0] + 1) * 127.5), 0, 255).astype(np.uint8)
    ).convert("RGB")


def evaluate(root):
    plan = check(root)
    data = load_data(root, plan)
    models = {}
    for name in METHODS:
        m, _ = make_model(root)
        load_weights(m, root / f"snapshots/step-{STEPS}" / name / "raw.npz")
        models[name] = m
    out = root / "outputs"
    out.mkdir(exist_ok=False)
    records = []
    for r in plan["records"]:
        name = r["name"]
        d = data[name]
        n = r["size"]
        arrays = {}
        for depth, partial in [(x, False) for x in (1, 3, 6)] + (
            [(3, True)] if name in PARTIAL else []
        ):
            mask = strength(n, partial)
            p = prepare_inputs(
                jnp.asarray(d["graph_oracle"][None]),
                jnp.asarray(d["graph_source"][None]),
                jnp.asarray(halo(mask, 0 if partial else 0.5)[None]),
                jax.random.PRNGKey(SEED + 1),
                depth,
            )
            suffix = f"d{depth}" + ("-partial" if partial else "")
            for method, m in models.items():
                h = np.asarray(infer_heads(m, p)[0])[0]
                assert np.isfinite(h).all()
                image, stats = decode(h)
                raw = crop(image)
                image = np.where(mask > 0, raw, d["source"])
                np.testing.assert_array_equal(image[mask == 0], d["source"][mask == 0])
                arrays[f"{method}-head-{suffix}"] = h
                arrays[f"{method}-{suffix}"] = image
                records.append(
                    dict(
                        name=name,
                        method=method,
                        depth=depth,
                        partial=partial,
                        training=r["training"],
                        graph=stats,
                        metrics=graph_metrics(crop(h), crop(d["target0"]), raw, d["oracle"]),
                        protected_exact=True,
                    )
                )
        np.savez_compressed(out / f"{name}.npz", **arrays)
    write(
        root / "evaluation.json",
        dict(count=len(records), records=records, outputs={p.name: sha(p) for p in out.iterdir()}),
    )
    pages(root, plan, data)


def pages(root, plan, data):
    out = root / "native"
    out.mkdir(exist_ok=False)
    manifest = []

    def page(filename, names, kind, method=None):
        n = data[names[0]]["source"].shape[0]
        canvas = Image.new("RGB", (n * 4, 24 + (n + 20) * len(names)), "#ddd")
        draw = ImageDraw.Draw(canvas)
        labels = (
            ["source", "HR initial", "HS control", "HS sparse"]
            if kind != "depth"
            else ["source", method + " d1", method + " d3", method + " d6"]
        )
        for i, label in enumerate(labels):
            draw.text((i * n + 3, 4), label, fill="black")
        for row, name in enumerate(names):
            with (
                np.load(root / "outputs" / f"{name}.npz") as z,
                np.load(HR / "outputs" / f"{name}.npz") as old,
            ):
                suffix = "d3-partial" if kind == "partial" else "d3"
                imgs = [data[name]["source"]] + (
                    [z[f"{method}-d{d}"] for d in (1, 3, 6)]
                    if kind == "depth"
                    else [old[f"graph-{suffix}"], z[f"control-{suffix}"], z[f"sparse-{suffix}"]]
                )
                for col, a in enumerate(imgs):
                    canvas.paste(to_image(a), (col * n, 44 + row * (n + 20)))
            draw.text(
                (3, 24 + row * (n + 20)),
                name + (" FIT" if name in TRAIN else " TRANSFER"),
                fill="black",
            )
        path = out / f"{filename}.png"
        canvas.save(path)
        manifest.append(
            dict(
                file=path.name,
                sha256=sha(path),
                names=names,
                kind=kind,
                method=method,
                native_pixels=True,
            )
        )

    for kind, methods in [("main", [None]), ("depth", METHODS)]:
        for method in methods:
            for group, per in [("probe", 4), ("scene", 2)]:
                names = [r["name"] for r in plan["records"] if r["name"].startswith(group)]
                for i in range(0, len(names), per):
                    page(
                        f"{kind}-{method or 'comparison'}-{group}-{i // per}",
                        names[i : i + per],
                        kind,
                        method,
                    )
    for group in ("probe", "scene"):
        names = [n for n in PARTIAL if n.startswith(group)]
        for i in range(0, len(names), 2):
            page(f"partial-{group}-{i // 2}", names[i : i + 2], "partial")
    assert len(manifest) == 31
    write(out / "manifest.json", manifest)


def finalize(root):
    plan = check(root)
    assert read(root / "training-summary.json")["steps"] == STEPS
    e = read(root / "evaluation.json")
    assert e["count"] == 166
    files = [
        "plan.json",
        "preflight.json",
        "prefix-comparison.json",
        "training-summary.json",
        "training/metrics.jsonl",
        "training/sampling.jsonl",
        "evaluation.json",
        "tests.json",
        "decision.json",
    ]
    for f, h in e["outputs"].items():
        assert sha(root / "outputs" / f) == h
        files.append("outputs/" + f)
    for group, count in [("native", 31), ("oracle", 8)]:
        m = read(root / group / "manifest.json")
        notes = read(root / group / "observations.json")
        assert len(m) == count
        assert set(notes) == {r["file"] for r in m} and all(notes.values())
        files += [f"{group}/manifest.json", f"{group}/observations.json"]
        for r in m:
            assert sha(root / group / r["file"]) == r["sha256"]
            files.append(f"{group}/" + r["file"])
    files += (
        ["oracle/metrics.json"]
        + [str(p.relative_to(root)) for p in sorted((root / "oracle").glob("*.npz"))]
        + [str(p.relative_to(root)) for p in sorted((root / "snapshots").rglob("*.npz"))]
    )
    assert read(root / "tests.json")["passed"] and not read(root / "decision.json")["adoption"]
    write(
        root / "completion.json",
        dict(
            stage="complete_sparse_graph_loss_pilot",
            commit=plan["commit"],
            steps_per_method=STEPS,
            methods=METHODS,
            native_pages_reviewed=39,
            new_conditions=166,
            adoption=False,
            fresh_used=False,
            evidence={f: sha(root / f) for f in files},
        ),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "phase",
        choices=["prepare", "oracle_diagnostic", "preflight", "train", "evaluate", "finalize"],
    )
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    start = time.monotonic()
    begin = now()
    globals()[a.phase](a.root)
    print(
        json.dumps(
            dict(
                phase=a.phase, started_at=begin, finished_at=now(), seconds=time.monotonic() - start
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
