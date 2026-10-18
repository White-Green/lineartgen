"""HR fixed graph-learning pilot; narrow fitting and known transfer are separate."""

import argparse
import datetime
import json
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from campaign_protocol import read, sha, write
from centerline_grid import encode, render
from cue_capacity import make_arm, prepare_inputs
from flax import nnx
from learned_centerline_model import (
    GraphCNN,
    convert,
    decode,
    gradient,
    infer_heads,
    load_weights,
    save_weights,
    target,
)
from learned_coarse_axis_model import infer as field_infer
from learned_coarse_axis_model import load_field
from opaque_geometry import geometry
from PIL import Image, ImageDraw
from plausible_study import digest, exact
from scipy.ndimage import binary_dilation, gaussian_filter

from lineartgen.config import load_config, save_config
from lineartgen.training import checkpoints, lineart_state

HQ = Path(
    "../jax-centerline-grid-audit-20261018/runs/centerline-grid-audit-20261018-retry1"
).resolve()
HP = Path("../jax-support-margin-cleanup-20261018/runs/support-margin-cleanup-20261018").resolve()
DQ_PARENT = Path("../jax-coherent-path-20261014/runs/coherent-path-20261014").resolve()
DOC = Path("docs/learned-centerline-plan-20261018.md")
CODE = tuple(
    map(
        Path,
        [
            "scripts/learned_centerline_model.py",
            "scripts/learned_centerline_study.py",
            "tests/test_learned_centerline.py",
        ],
    )
)
TRAIN = tuple(f"probe-{i:02d}" for i in (0, 12, 18, 24, 36, 42, 48, 54, 56, 58))
PARTIAL = tuple(f"probe-{i:02d}" for i in (24, 36, 42, 48, 56, 62)) + ("scene-2", "scene-7")
STEPS = 1200
SAVED = (0, 6, 400, 800, 1200)
SEED = 2026101807


def now():
    return datetime.datetime.now(datetime.UTC).isoformat()


def make_model(root):
    cfg = load_config(root / "config.json")
    model = GraphCNN(cfg.model, rngs=nnx.Rngs(SEED))
    load_weights(model, root / "initial.npz")
    return model, cfg


def check(root):
    p = read(root / "plan.json")
    for f, h in p["bindings"].items():
        assert sha(Path(f)) == h, f
    assert sha(DOC) == p["document_sha256"]
    for f, h in p["code"].items():
        assert sha(Path(f)) == h
    assert sha(root / "source.tar.gz") == p["archive_sha256"]
    return p


def logits_from_target(t):
    h = np.zeros_like(t)
    h[..., [0, 4, 5, 6, 7, 8]] = np.where(t[..., [0, 4, 5, 6, 7, 8]] > 0, 20, -20)
    p = np.clip(t[..., 1:3], 1e-6, 1 - 1e-6)
    h[..., 1:3] = np.log(p / (1 - p))
    h[..., 3] = t[..., 3]
    return h


def prepare(root):
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    assert not root.exists()
    root.mkdir(parents=True)
    for ref in (HQ, HP):
        for f, h in read(ref / "completion.json")["evidence"].items():
            assert sha(ref / f) == h
    previous, cfg = load_field(HP / "snapshots/step-1200/consistent/raw")
    cfg = cfg.model_copy(
        update=dict(
            train=cfg.train.model_copy(
                update=dict(
                    steps=STEPS,
                    learning_rate_decay_steps=STEPS,
                    learning_rate=1e-4,
                    minimum_learning_rate=2e-5,
                    seed=SEED,
                )
            )
        )
    )
    assert cfg.train.batch_size == 4 and cfg.train.peak_learning_rate == 2e-4
    model = convert(previous, cfg)
    save_config(cfg, root / "config.json")
    save_weights(model, root / "initial.npz")
    restored, _ = make_model(root)
    exact(nnx.state(model, nnx.Param), nnx.state(restored, nnx.Param))
    records = []
    inputs = []
    external = [HP / "completion.json", HQ / "completion.json"] + list(
        (HP / "snapshots/step-1200/consistent/raw").iterdir()
    )
    for r in read(HQ / "plan.json")["records"]:
        if r["group"] == "fixture":
            continue
        p = HQ / r["file"]
        external.append(p)
        with np.load(p) as z:
            inputs.append(
                (r["name"], {k: z[k] for k in ("segments", "widths", "rectangles", "source")})
            )
    meta = read(DQ_PARENT / "data.json")
    row = next(r for r in meta["evaluation"] if r["split"] == "new_probe" and r["index"] == 37)
    sourcefile = DQ_PARENT / row["file"]
    external += [DQ_PARENT / "data.json", sourcefile]
    with np.load(sourcefile) as z:
        source = z["source"]
    s, w, r = geometry(row["metadata"], kind=row["kind"])
    inputs.append(("probe-37", dict(segments=s, widths=w, rectangles=r, source=source)))
    inputs.sort(key=lambda x: x[0])
    audit = []
    for name, data in inputs:
        n = data["source"].shape[0]
        packed = dict(data)
        stages = []
        for level in range(6):
            t, stats = target(data["segments"], data["widths"], data["rectangles"], n, level)
            packed[f"target{level}"] = t
            stages.append(stats)
        oracle, st = decode(logits_from_target(packed["target0"]))
        g = encode(data["segments"], data["widths"], 1)
        reference = render(g["segments"], g["segment_widths"], data["rectangles"], n)
        error = float(np.abs(oracle - reference).max())
        np.testing.assert_allclose(oracle, reference, atol=1e-4, rtol=1e-4)
        packed["oracle"] = oracle
        p = root / "inputs" / f"{name}.npz"
        p.parent.mkdir(exist_ok=True)
        np.savez_compressed(p, **packed)
        records.append(
            dict(
                name=name,
                size=n,
                training=name in TRAIN,
                file=str(p.relative_to(root)),
                sha256=sha(p),
            )
        )
        audit.append(dict(name=name, stages=stages, roundtrip_max_error=error, decode_stats=st))
    assert len(records) == 25 and sum(r["training"] for r in records) == 10
    write(root / "target-audit.json", audit)
    files = (
        external
        + [root / "initial.npz", root / "config.json", root / "target-audit.json"]
        + [root / r["file"] for r in records]
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
            code={str(p): sha(p) for p in CODE},
            document_sha256=sha(DOC),
            archive_sha256=sha(root / "source.tar.gz"),
            parameters=sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param))),
            steps=STEPS,
            seed=SEED,
            native_pages=22,
            fresh_used=False,
            adoption=False,
        ),
    )


def load_data(root, plan):
    result = {}
    for r in plan["records"]:
        with np.load(root / r["file"]) as z:
            result[r["name"]] = {k: z[k] for k in z.files}
    return result


def strength(size, partial):
    yy, xx = np.mgrid[:size, :size]
    return (
        np.where(xx > 0.6 * size - 0.25 * yy, 0.5, 0).astype(np.float32)[..., None]
        if partial
        else np.full((size, size, 1), 0.5, np.float32)
    )


def batch(data, step):
    rng = np.random.default_rng([SEED, step])
    indices = rng.integers(len(TRAIN), size=4)
    partial = rng.random(4) < 0.25
    opacity = rng.uniform(0.65, 1, size=4)
    sigma = rng.uniform(0, 1, size=4)
    source = np.stack(
        [
            1
            - 2
            * opacity[k]
            * gaussian_filter(
                (1 - data[TRAIN[i]]["source"]) / 2, (sigma[k], sigma[k], 0), mode="constant", cval=0
            )
            for k, i in enumerate(indices)
        ]
    ).astype(np.float32)
    masks = np.stack([strength(256, v) for v in partial])
    depth = step % 6 + 1
    targets = tuple(
        jnp.asarray(np.stack([data[TRAIN[i]][f"target{level}"] for i in indices]))
        for level in range(depth)
    )
    clean = np.stack([data[TRAIN[i]]["oracle"] for i in indices])
    p = prepare_inputs(
        jnp.asarray(clean),
        jnp.asarray(source),
        jnp.asarray(masks),
        jax.random.fold_in(jax.random.PRNGKey(SEED), step),
        depth,
    )
    audit = dict(
        indices=indices.tolist(),
        partial=partial.tolist(),
        opacity=opacity.tolist(),
        sigma=sigma.tolist(),
        depth=depth,
        input_sha256=digest(p),
        target_sha256=digest(targets),
    )
    return p, targets, audit


def arm(root):
    m, c = make_model(root)
    a = make_arm(c, m)
    a["gradient"] = gradient
    return a


def update(a, p, t):
    parts, grads = gradient(a["model"], p, t)
    parts["gradient_norm"] = optax.global_norm(grads)
    values = {k: float(v) for k, v in parts.items()}
    assert all(np.isfinite(v) for v in values.values())
    _, a["ema"] = a["update"](a["model"], a["optimizer"], grads, a["ema"])
    jax.block_until_ready(a["ema"])
    return values


def preflight(root):
    plan = check(root)
    data = load_data(root, plan)
    a, b = arm(root), arm(root)
    rows = []
    for step in range(6):
        p, t, audit = batch(data, step)
        x, y = update(a, p, t), update(b, p, t)
        assert x == y
        exact(
            lineart_state.payload({"graph": a}, step + 1),
            lineart_state.payload({"graph": b}, step + 1),
        )
        rows.append(dict(step=step, metrics=x, **audit))
    assert all(r["metrics"]["gradient_norm"] > 0 for r in rows)
    write(root / "preflight.json", dict(steps=6, all_depths=True, all_state_exact=True, rows=rows))


def snapshot(root, a, step, manager):
    lineart_state.save(manager, {"graph": a}, step)
    for mode in ("raw", "ema"):
        m = nnx.clone(a["model"])
        if mode == "ema":
            nnx.update(m, a["ema"])
        p = root / "snapshots" / f"step-{step}" / f"{mode}.npz"
        save_weights(m, p)
        restored, _ = make_model(root)
        load_weights(restored, p)
        exact(nnx.state(m, nnx.Param), nnx.state(restored, nnx.Param))
    restored = arm(root)
    lineart_state.restore(manager, {"graph": restored}, step)
    exact(
        lineart_state.payload({"graph": a}, step), lineart_state.payload({"graph": restored}, step)
    )


def train(root):
    plan = check(root)
    data = load_data(root, plan)
    a = arm(root)
    assert read(root / "preflight.json")["all_state_exact"]
    directory = root / "training"
    directory.mkdir(exist_ok=False)
    rows = []
    with (
        checkpoints.manager(root / "checkpoints", keep=len(SAVED)) as manager,
        (directory / "metrics.jsonl").open("w") as log,
        (directory / "sampling.jsonl").open("w") as sample,
    ):
        snapshot(root, a, 0, manager)
        for step in range(STEPS):
            start = time.monotonic()
            p, t, audit = batch(data, step)
            values = update(a, p, t)
            row = dict(
                step=step + 1, depth=step % 6 + 1, seconds=time.monotonic() - start, **values
            )
            log.write(json.dumps(row) + "\n")
            log.flush()
            sample.write(json.dumps(dict(step=step, **audit)) + "\n")
            sample.flush()
            if step < 6:
                expected = read(root / "preflight.json")["rows"][step]
                assert audit == {k: v for k, v in expected.items() if k not in ("step", "metrics")}
                assert values == expected["metrics"]
            if step + 1 in SAVED:
                snapshot(root, a, step + 1, manager)
            rows.append(row)
    write(
        root / "training-summary.json",
        dict(
            steps=len(rows),
            depth_counts={str(d): sum(r["depth"] == d for r in rows) for d in range(1, 7)},
            saved=SAVED,
            all_saved_states_exact=True,
            all_snapshots_exact=True,
            first_six_exact=True,
            final=rows[-1],
            mean_seconds_last600=float(np.mean([r["seconds"] for r in rows[-600:]])),
        ),
    )


def graph_metrics(h, t, image, oracle):
    result = {}
    for name, sl in [("vertex", 0), ("edge", slice(4, 8)), ("fill", 8)]:
        p = h[..., sl] >= 0
        y = t[..., sl] > 0
        tp = int((p & y).sum())
        fp = int((p & ~y).sum())
        fn = int((~p & y).sum())
        result[name] = dict(
            tp=tp, fp=fp, fn=fn, precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1)
        )
    a = (1 - image[..., 0]) / 2
    y = (1 - oracle[..., 0]) / 2
    support = binary_dilation(y > 0.01, iterations=2)
    core = y > 0.8
    result["outside_alpha"] = float(a[~support].sum())
    result["core_alpha8_coverage"] = float((a[core] >= 0.8).mean()) if core.any() else None
    result["core_mean_alpha"] = float(a[core].mean()) if core.any() else None
    result["alpha_sum"] = float(a.sum())
    return result


def evaluate(root):
    plan = check(root)
    data = load_data(root, plan)
    model, cfg = make_model(root)
    load_weights(model, root / f"snapshots/step-{STEPS}/raw.npz")
    baseline, _ = load_field(HP / "snapshots/step-1200/consistent/raw")
    out = root / "outputs"
    out.mkdir(exist_ok=False)
    records = []
    for r in plan["records"]:
        name = r["name"]
        item = data[name]
        n = r["size"]
        arrays = {}
        modes = [(d, False) for d in (1, 3, 6)] + ([(3, True)] if name in PARTIAL else [])
        for depth, partial in modes:
            mask = strength(n, partial)
            p = prepare_inputs(
                jnp.asarray(item["oracle"][None]),
                jnp.asarray(item["source"][None]),
                jnp.asarray(mask[None]),
                jax.random.PRNGKey(SEED + 1),
                depth,
            )
            h = np.asarray(infer_heads(model, p)[0])[0]
            assert np.isfinite(h).all()
            raw, stats = decode(h)
            result = np.where(mask > 0, raw, item["source"])
            control = np.asarray(field_infer(baseline, p))[0]
            np.testing.assert_array_equal(result[mask == 0], item["source"][mask == 0])
            np.testing.assert_array_equal(control[mask == 0], item["source"][mask == 0])
            suffix = f"d{depth}" + ("-partial" if partial else "")
            arrays.update({f"head-{suffix}": h, f"graph-{suffix}": result, f"hp-{suffix}": control})
            records.append(
                dict(
                    name=name,
                    depth=depth,
                    partial=partial,
                    training=r["training"],
                    graph=stats,
                    metrics=graph_metrics(h, item["target0"], raw, item["oracle"]),
                    protected_exact=True,
                )
            )
        path = out / f"{name}.npz"
        np.savez_compressed(path, **arrays)
    write(
        root / "evaluation.json",
        dict(
            records=records,
            outputs={p.name: sha(p) for p in out.iterdir()},
            count=len(records),
            fresh_used=False,
        ),
    )
    pages(root, plan, data)


def pages(root, plan, data):
    dest = root / "native"
    dest.mkdir(exist_ok=False)
    manifest = []

    def page(name, names, kind):
        size = data[names[0]]["source"].shape[0]
        top = 24
        pad = 20
        canvas = Image.new("RGB", (size * 4, (size + pad) * len(names) + top), "#ddd")
        draw = ImageDraw.Draw(canvas)
        labels = {
            "main": ["source", "known graph oracle", "HP d3", "HR d3"],
            "depth": ["source", "HR d1", "HR d3", "HR d6"],
            "partial": ["source", "active area", "HP partial d3", "HR partial d3"],
        }[kind]
        for col, label in enumerate(labels):
            draw.text((col * size + 3, 4), label, fill="black")
        for row, key in enumerate(names):
            d = data[key]
            with np.load(root / "outputs" / f"{key}.npz") as z:
                if kind == "main":
                    images = [d["source"], d["oracle"], z["hp-d3"], z["graph-d3"]]
                elif kind == "depth":
                    images = [d["source"], z["graph-d1"], z["graph-d3"], z["graph-d6"]]
                else:
                    images = [
                        d["source"],
                        1 - 4 * strength(size, True),
                        z["hp-d3-partial"],
                        z["graph-d3-partial"],
                    ]
                for col, v in enumerate(images):
                    im = Image.fromarray(
                        np.clip(np.rint((v[..., 0] + 1) * 127.5), 0, 255).astype(np.uint8)
                    ).convert("RGB")
                    canvas.paste(im, (col * size, top + row * (size + pad) + pad))
            draw.text(
                (3, top + row * (size + pad)),
                key + (" FIT" if key in TRAIN else " TRANSFER"),
                fill="black",
            )
        p = dest / f"{name}.png"
        canvas.save(p)
        manifest.append(
            dict(file=p.name, sha256=sha(p), names=names, kind=kind, native_pixels=True)
        )

    for kind in ("main", "depth"):
        for group, per in [("probe", 4), ("scene", 2)]:
            names = [r["name"] for r in plan["records"] if r["name"].startswith(group)]
            for i in range(0, len(names), per):
                page(f"{kind}-{group}-{i // per}", names[i : i + per], kind)
    for group, per in [("probe", 2), ("scene", 2)]:
        names = [n for n in PARTIAL if n.startswith(group)]
        for i in range(0, len(names), per):
            page(f"partial-{group}-{i // per}", names[i : i + per], "partial")
    assert len(manifest) == 22
    write(dest / "manifest.json", manifest)


def finalize(root):
    plan = check(root)
    assert read(root / "training-summary.json")["steps"] == STEPS
    e = read(root / "evaluation.json")
    assert e["count"] == 83
    for f, h in e["outputs"].items():
        assert sha(root / "outputs" / f) == h
    manifest = read(root / "native/manifest.json")
    notes = read(root / "native/observations.json")
    assert set(notes) == {r["file"] for r in manifest} and all(notes.values())
    for r in manifest:
        assert sha(root / "native" / r["file"]) == r["sha256"]
    assert read(root / "tests.json")["passed"]
    assert read(root / "decision.json")["adoption"] is False
    files = (
        [
            root / "plan.json",
            root / "target-audit.json",
            root / "preflight.json",
            root / "training-summary.json",
            root / "training/metrics.jsonl",
            root / "training/sampling.jsonl",
            root / "evaluation.json",
            root / "native/manifest.json",
            root / "native/observations.json",
            root / "decision.json",
            root / "tests.json",
        ]
        + [root / "native" / r["file"] for r in manifest]
        + [root / "outputs" / f for f in e["outputs"]]
        + sorted((root / "snapshots").rglob("*.npz"))
    )
    write(
        root / "completion.json",
        dict(
            stage="complete_learned_centerline_pilot",
            commit=plan["commit"],
            steps=STEPS,
            native_pages_reviewed=22,
            output_conditions=83,
            adoption=False,
            fresh_used=False,
            evidence={str(p.relative_to(root)): sha(p) for p in files},
        ),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["prepare", "preflight", "train", "evaluate", "finalize"])
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    start = time.monotonic()
    started = now()
    globals()[args.phase](args.root)
    print(
        json.dumps(
            dict(
                phase=args.phase,
                started_at=started,
                finished_at=now(),
                seconds=time.monotonic() - start,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
