"""HY: matched broad synthetic training of narrow/wide attraction graphs."""

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
from attraction_augmentation import spatial, target_action
from attraction_graph import decode, logits_from_target
from attraction_radius_objective import narrow_gradient as sparse_gradient
from attraction_radius_objective import wide_gradient as control_gradient
from campaign_protocol import read, sha, write
from cue_capacity import make_arm, prepare_inputs
from flax import nnx
from learned_centerline_model import (
    infer_heads,
    load_weights,
    save_weights,
)
from learned_centerline_study import (
    PARTIAL,
    SEED,
    crop,
    digest,
    graph_metrics,
    halo,
    load_data,
    make_model,
    strength,
)
from PIL import Image, ImageDraw
from plausible_study import exact
from projected_edge_gate import gate
from scipy.ndimage import gaussian_filter

from lineartgen.training import checkpoints, lineart_state

HX = Path("../jax-attraction-radius-20261018/runs/attraction-radius-20261018").resolve()
STEPS = 6000
SAVED = (0, 6, 1200, 3000, 6000)
DATA_SEED = 2026101811
CACHE = Path("/tmp/lineart-graph-training-cache-20261018")
LABEL_CACHE = {
    "control": Path("/tmp/lineart-attraction-training-cache-20261018"),
    "sparse": Path("/tmp/lineart-attraction-radius1-cache-20261018"),
}
DOC = Path("docs/broad-attraction-plan-20261018.md")
CODE = tuple(
    map(
        Path,
        [
            "scripts/broad_attraction_study.py",
            "scripts/learned_centerline_study.py",
            "scripts/learned_centerline_model.py",
            "scripts/cue_capacity.py",
            "tests/test_broad_attraction.py",
            "scripts/attraction_radius_objective.py",
            "scripts/attraction_augmentation.py",
            "scripts/projected_edge_gate.py",
            "tests/test_attraction_radius.py",
            "scripts/attraction_graph.py",
            "scripts/attraction_objective.py",
            "tests/test_attraction_objective.py",
        ],
    )
)
METHODS = ("control", "sparse")
RADII = {"control": 4.0, "sparse": 1.0}


def now():
    return datetime.datetime.now(datetime.UTC).isoformat()


def prepare(root):
    assert not root.exists()
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    for f, h in read(HX / "completion.json")["evidence"].items():
        assert sha(HX / f) == h
    root.mkdir(parents=True)
    cfg = read(HX / "config.json")
    cfg["train"].update(steps=STEPS, learning_rate_decay_steps=STEPS)
    write(root / "config.json", cfg)
    shutil.copy2(HX / "initial.npz", root / "initial.npz")
    model, c = make_model(root)
    assert c.train.peak_learning_rate == 2e-4 and c.train.steps == STEPS
    assert sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param))) == 22725
    (root / "inputs").mkdir()
    records = []
    eval_hashes = {}
    for r in read(HX / "plan.json")["records"]:
        dest = root / r["file"]
        shutil.copy2(HX / r["file"], dest)
        with np.load(dest) as z:
            eval_hashes[digest(z["source"])] = r["name"]
        records.append(dict(r, training=r["name"] in ("probe-54", "probe-56"), sha256=sha(dest)))
    datasets = {"source": CACHE, **LABEL_CACHE}
    manifests = {k: read(v / "manifest.json") for k, v in datasets.items()}
    assert all(len(m["records"]) == 512 for m in manifests.values())
    training = []
    files = [
        HX / "completion.json",
        HX / "supplemental-binding.json",
        HX / "initial.npz",
        root / "initial.npz",
        root / "config.json",
    ]
    files += [root / r["file"] for r in records]
    files += [HX / "outputs" / (r["name"] + ".npz") for r in records]
    for directory in datasets.values():
        files += [directory / name for name in ("plan.json", "manifest.json", "recipe.py")]
    kinds = {}
    overlaps = []
    for i in range(512):
        row = {}
        for k, directory in datasets.items():
            r = manifests[k]["records"][i]
            assert r["index"] == i
            p = directory / r["file"]
            assert sha(p) == r["sha256"]
            row[k] = str(p)
            files.append(p)
        r = manifests["source"]["records"][i]
        row.update(index=i, kind=r["kind"])
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        with np.load(row["source"]) as z:
            match = eval_hashes.get(digest(z["source"]))
            if match is not None:
                assert r["kind"] in ("blank", "black") and match in ("probe-54", "probe-56")
                overlaps.append(dict(index=i, kind=r["kind"], evaluation=match))
        for k in METHODS:
            stages = manifests[k]["records"][i]["stages"]
            assert all(not st["unrepresented_paths"] for st in stages[:5])
        training.append(row)
    write(
        root / "training-data.json",
        dict(
            records=training,
            kinds=kinds,
            count=512,
            evaluation_source_overlap=overlaps,
            coarse_level5_missing_indices=[251, 270],
            geometric_warp=False,
        ),
    )
    files += [root / "training-data.json"]
    data_teacher_pages(root, training)
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
            radii=RADII,
            steps=STEPS,
            native_pages=31,
            teacher_pages=12,
            training_examples=512,
            comparison_tolerance=dict(atol=1e-5, rtol=1e-5),
            fresh_used=False,
            adoption=False,
        ),
    )


def load_training(root):
    data = []
    for r in read(root / "training-data.json")["records"]:
        with np.load(r["source"]) as z:
            d = {k: z[k] for k in ("source", "graph_oracle")}
        for method, prefix in [("control", "target"), ("sparse", "smalltarget")]:
            with np.load(r[method]) as z:
                d.update({f"{prefix}{level}": z[f"target{level}"] for level in range(6)})
        data.append(d)
    return data


def data_teacher_pages(root, records):
    out = root / "teacher"
    out.mkdir()
    chosen = {}
    for r in records:
        chosen.setdefault(r["kind"], r)
    manifest = []
    for kind, r in chosen.items():
        with np.load(r["source"]) as z:
            source = z["source"]
        images = [source]
        for method in METHODS:
            with np.load(r[method]) as z:
                t = z["target0"]
            image, _ = decode(
                logits_from_target(t, base_radius=RADII[method]), base_radius=RADII[method]
            )
            images.append(crop(image))
        canvas = Image.new("RGB", (768, 284), "#ddd")
        draw = ImageDraw.Draw(canvas)
        for col, (label, image) in enumerate(
            zip(("source", "R4 teacher", "R1 teacher"), images, strict=True)
        ):
            draw.text((col * 256 + 3, 4), f"{kind} {r['index']} {label}", fill="black")
            canvas.paste(to_image(image), (col * 256, 28))
        p = out / f"{kind}.png"
        canvas.save(p)
        manifest.append(dict(file=p.name, sha256=sha(p), index=r["index"], native_pixels=True))
    assert len(manifest) == 12
    write(out / "manifest.json", manifest)


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
        a["target_key"] = name
        result[name] = a
    return result


def update(a, p, t):
    parts, g = a["gradient"](a["model"], p, t[a["target_key"]])
    parts["gradient_norm"] = optax.global_norm(g)
    values = {k: float(v) for k, v in parts.items()}
    assert all(np.isfinite(v) for v in values.values())
    _, a["ema"] = a["update"](a["model"], a["optimizer"], g, a["ema"])
    jax.block_until_ready(a["ema"])
    return values


def augment_prepared(prepared, groups):
    def apply(a):
        return jnp.asarray(
            np.stack([spatial(np.asarray(x), int(g)) for x, g in zip(a, groups, strict=True)])
        )

    return jax.tree.map(apply, prepared)


def sampling(step, count):
    rng = np.random.default_rng([DATA_SEED, step])
    return dict(
        indices=rng.integers(count, size=4).tolist(),
        partial=(rng.random(4) < 0.25).tolist(),
        opacity=rng.uniform(0.65, 1, size=4).tolist(),
        sigma=rng.uniform(0, 1, size=4).tolist(),
        groups=rng.integers(8, size=4).tolist(),
        depth=step % 6 + 1,
    )


def matched_batch(data, step):
    audit = sampling(step, len(data))
    source = np.stack(
        [
            1
            - 2
            * audit["opacity"][k]
            * gaussian_filter(
                (1 - data[i]["source"]) / 2,
                (audit["sigma"][k], audit["sigma"][k], 0),
                mode="constant",
                cval=0,
            )
            for k, i in enumerate(audit["indices"])
        ]
    ).astype(np.float32)
    source = np.stack([halo(v) for v in source])
    masks = np.stack([halo(strength(256, v), 0 if v else 0.5) for v in audit["partial"]])
    clean = np.stack([data[i]["graph_oracle"] for i in audit["indices"]])
    p = prepare_inputs(
        jnp.asarray(clean),
        jnp.asarray(source),
        jnp.asarray(masks),
        jax.random.fold_in(jax.random.PRNGKey(DATA_SEED), step),
        audit["depth"],
    )
    groups = audit["groups"]
    targets = {}
    for name, prefix in [("control", "target"), ("sparse", "smalltarget")]:
        targets[name] = tuple(
            jnp.asarray(
                np.stack(
                    [
                        target_action(data[i][f"{prefix}{level}"], g)
                        for i, g in zip(audit["indices"], groups, strict=True)
                    ]
                )
            )
            for level in range(audit["depth"])
        )
    p = augment_prepared(p, groups)
    audit.update(input_sha256=digest(p), target_sha256={k: digest(v) for k, v in targets.items()})
    return p, targets, audit


def preflight(root):
    check(root)
    data = load_training(root)
    a, b = arms(root), arms(root)
    rows = []
    for step in range(6):
        p, t, audit = matched_batch(data, step)
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
            assert comparisons[name]["metrics"]["bit_exact"]
            assert comparisons[name]["state"]["bit_exact"]
            metrics[name] = x
        rows.append(dict(step=step, metrics=metrics, comparisons=comparisons, **audit))
    write(root / "preflight.json", dict(steps=6, rows=rows, all_comparisons_passed=True))
    p, _, _ = matched_batch(data, 1)
    replaced = p._replace(targets=tuple(jnp.full_like(t, -999) for t in p.targets))
    for name in METHODS:
        exact(infer_heads(a[name]["model"], p), infer_heads(a[name]["model"], replaced))
    write(
        root / "target-isolation-audit.json",
        dict(all_heads_exact=True, methods=METHODS, nonconstant_weights=True, depth=2),
    )


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
    check(root)
    assert read(root / "teacher/decision.json")["passed"]
    data = load_training(root)
    models = arms(root)
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
            p, t, audit = matched_batch(data, step)
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
            same_input_both_methods=True,
            training_examples=512,
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
                gated, rejected = gate(h, base_radius=RADII[method])
                image, stats = decode(gated, base_radius=RADII[method])
                stats["rejected_edges"] = rejected
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
                        metrics=graph_metrics(
                            crop(h),
                            crop(d["target0" if method == "control" else "smalltarget0"]),
                            raw,
                            d["oracle" if method == "control" else "smalloracle"],
                        ),
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
            ["source", "HX R1 ten-shape pilot", "HY R4 broad", "HY R1 broad"]
            if kind != "depth"
            else ["source", method + " d1", method + " d3", method + " d6"]
        )
        for i, label in enumerate(labels):
            draw.text((i * n + 3, 4), label, fill="black")
        for row, name in enumerate(names):
            with (
                np.load(root / "outputs" / f"{name}.npz") as z,
                np.load(HX / "outputs" / f"{name}.npz") as old,
            ):
                suffix = "d3-partial" if kind == "partial" else "d3"
                imgs = [data[name]["source"]] + (
                    [z[f"{method}-d{d}"] for d in (1, 3, 6)]
                    if kind == "depth"
                    else [old[f"sparse-{suffix}"], z[f"control-{suffix}"], z[f"sparse-{suffix}"]]
                )
                for col, a in enumerate(imgs):
                    canvas.paste(to_image(a), (col * n, 44 + row * (n + 20)))
            draw.text(
                (3, 24 + row * (n + 20)),
                name
                + (
                    " TRAINING BLANK/FILL"
                    if name in ("probe-54", "probe-56")
                    else " KNOWN EVALUATION"
                ),
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
    for group, count in [("native", 31), ("teacher", 12)]:
        m = read(root / group / "manifest.json")
        notes = read(root / group / "observations.json")
        assert len(m) == count
        assert set(notes) == {r["file"] for r in m} and all(notes.values())
        files += [f"{group}/manifest.json", f"{group}/observations.json"]
        for r in m:
            assert sha(root / group / r["file"]) == r["sha256"]
            files.append(f"{group}/" + r["file"])
    files += [str(p.relative_to(root)) for p in sorted((root / "snapshots").rglob("*.npz"))]
    files += ["training-data.json", "target-isolation-audit.json", "teacher/decision.json"]
    assert read(root / "tests.json")["passed"] and not read(root / "decision.json")["adoption"]
    write(
        root / "completion.json",
        dict(
            stage="complete_broad_attraction_comparison",
            commit=plan["commit"],
            steps_per_method=STEPS,
            methods=METHODS,
            radii=RADII,
            native_pages_reviewed=43,
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
        choices=["prepare", "preflight", "train", "evaluate", "finalize"],
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
