"""Fixed512px weights: project predicted endpoints before each Euler advance."""

import argparse
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from bridge_time_grid import time_nodes
from campaign_protocol import read, sha, write
from diagnose_learnability import verify
from geometry_velocity_fit import EVAL_SEED, PROBE, TRAIN, arrays
from paper_band_fit import fit_checks
from PIL import Image, ImageDraw
from velocity_bridge_toy import initialize, velocity

from lineartgen.coverage_metrics import coverage_metrics
from lineartgen.evaluation import probe_metrics, protected_metrics
from lineartgen.lineart_metrics import halo_metrics
from lineartgen.models.serialization import load_inference

METHODS = ("mse", "interval")
DOC = Path("docs/projected-endpoint-plan-20261011.md")


@jax.jit
def advance(x, v, strength, source, remaining, delta):
    endpoint = x + remaining * v
    bounded_v = v + (jnp.clip(endpoint, -1, 1) - endpoint) / remaining
    return jnp.where(strength > 0, x + delta * bounded_v, source)


def sample(predict, source, strength, key, *, depth, steps, projected, trace=False):
    images, levels, guides = initialize(source, strength, key, depth)
    nodes = time_nodes(steps)
    records = []
    for index, (start, end) in enumerate(zip(nodes[:-1], nodes[1:], strict=True)):
        progress = tuple(jnp.full_like(x, 1 - start) for x in images)
        estimates = predict(images, levels, guides, progress)
        if trace:
            x, v = images[0], estimates[0]
            endpoint = x + (1 - start) * v
            records.append(
                dict(
                    index=index,
                    tau=float(start),
                    minimum=float(x.min()),
                    maximum=float(x.max()),
                    outside_fraction=float(jnp.mean(jnp.abs(x) > 1)),
                    endpoint_outside_fraction=float(jnp.mean(jnp.abs(endpoint) > 1)),
                    endpoint_minimum=float(endpoint.min()),
                    endpoint_maximum=float(endpoint.max()),
                )
            )
        if projected:
            images = tuple(
                advance(x, v, t, s, 1 - start, end - start)
                for x, v, t, s in zip(images, estimates, levels, guides, strict=True)
            )
        else:
            images = tuple(
                jnp.where(t > 0, x + (end - start) * v, s)
                for x, v, t, s in zip(images, estimates, levels, guides, strict=True)
            )
    return np.asarray(images[0]), records


def prepare(root):
    assert not root.exists()
    (root / "configs").mkdir(parents=True)
    parent = Path(
        "../jax-geometry-velocity-fit-20261010/runs/geometry-velocity-fit-20261010"
    ).resolve()
    verify(parent)
    assert read(parent / "final-audit.json")["steps_per_arm"] == 6000
    for m in METHODS:
        shutil.copytree(parent / f"snapshots/step-6000/{m}/raw", root / "initial" / m)
        shutil.copyfile(parent / f"configs/{m}.json", root / f"configs/{m}.json")
    for name in ("inputs.npz", "train.json", "probe.json"):
        shutil.copyfile(parent / name, root / name)
    initial = root / "initial"
    write(
        root / "plan.json",
        dict(
            initial=str(initial),
            initial_files={
                str(p.relative_to(initial)): sha(p) for p in initial.rglob("*") if p.is_file()
            },
            data={
                "fixed_pairs": dict(path=str(root / "inputs.npz"), sha256=sha(root / "inputs.npz"))
            },
            parent=str(parent),
            parent_files={
                n: sha(parent / n)
                for n in ("source.json", "final-audit.json", "evaluation/metrics.json")
            },
            crops=read(parent / "plan.json")["crops"],
            updates=0,
            steps=32,
            seed=EVAL_SEED,
            plan_document_sha256=sha(DOC),
            adoption=False,
        ),
    )


def check(root):
    verify(root)
    plan = read(root / "plan.json")
    assert sha(DOC) == plan["plan_document_sha256"]
    parent = Path(plan["parent"])
    verify(parent)
    for p, digest in plan["parent_files"].items():
        assert sha(parent / p) == digest
    return plan


def evaluate(root):
    plan = check(root)
    assert read(root / "test-results.json")["passed"]
    parent = Path(plan["parent"])
    old = read(parent / "evaluation/metrics.json")
    bindings = dict(old["bindings"])
    for path, digest in bindings.items():
        assert sha(Path(path)) == digest
    models = {m: load_inference(root / "initial" / m)[0] for m in METHODS}
    initial = load_inference(parent / "initial")[0]
    out = root / "evaluation"
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    rows = list(old["rows"])
    replay, traces = [], []
    with np.load(root / "inputs.npz") as data:
        for split, ids in (("train", TRAIN), ("probe", PROBE)):
            clean, source, strength = arrays(root, ids)
            for depth in (1, 3, 6):
                for draw in range(4):
                    key = jax.random.fold_in(
                        jax.random.PRNGKey(EVAL_SEED),
                        depth * 100 + int(split == "probe") * 10 + draw,
                    )
                    active = dict(models)
                    if depth == 3 and draw == 0:
                        active["initial"] = initial
                    for m, model in active.items():
                        for projected in (False, True) if m in METHODS else (False,):
                            physical, trace = sample(
                                lambda x, t, s, u, model=model: velocity(model, x, t, s, u),
                                source,
                                strength,
                                key,
                                depth=depth,
                                steps=32,
                                projected=projected,
                                trace=depth == 3 and draw == 0 and m in METHODS,
                            )
                            label = f"{m}_projected" if projected else m
                            prediction = np.clip(physical, -1, 1)
                            assert np.isfinite(physical).all()
                            if trace:
                                traces.append(dict(method=label, split=split, rows=trace))
                            if not projected:
                                path = (
                                    parent
                                    / f"evaluation/predictions/{split}-d{depth}-r{draw}-{m}.npz"
                                )
                                with np.load(path) as saved:
                                    np.testing.assert_array_equal(physical, saved["physical"])
                                    np.testing.assert_array_equal(prediction, saved["prediction"])
                                replay.append(str(path))
                                continue
                            path = out / f"predictions/{split}-d{depth}-r{draw}-{label}.npz"
                            np.savez_compressed(path, physical=physical, prediction=prediction)
                            bindings[str(path)] = sha(path)
                            for i, value in zip(ids, prediction, strict=True):
                                target, level = data[f"{i}-image"], data[f"{i}-strength"]
                                metrics = protected_metrics(value, target, level)
                                metrics |= halo_metrics(value, target, level)
                                metrics |= probe_metrics(
                                    value, target, data[f"{i}-probes"], data[f"{i}-connections"]
                                )
                                metrics |= coverage_metrics(value, target, level)
                                assert all(v is None or np.isfinite(v) for v in metrics.values())
                                rows.append(
                                    dict(
                                        method=label,
                                        case=i,
                                        split=split,
                                        depth=depth,
                                        draw=draw,
                                        **metrics,
                                    )
                                )
                    write(
                        out / "progress.json",
                        dict(split=split, depth=depth, draw=draw, rows=len(rows), total=392),
                    )
        clean, source, strength = arrays(root, TRAIN)
        strength = strength.at[:, :, :256].set(0)
        key = jax.random.fold_in(jax.random.PRNGKey(EVAL_SEED), 9000)
        for m, model in models.items():
            for projected in (False, True):
                physical, _ = sample(
                    lambda x, t, s, u, model=model: velocity(model, x, t, s, u),
                    source,
                    strength,
                    key,
                    depth=6,
                    steps=32,
                    projected=projected,
                )
                prediction = np.clip(physical, -1, 1)
                mask = np.asarray(strength) == 0
                assert np.isfinite(physical).all()
                np.testing.assert_array_equal(physical[mask], np.asarray(source)[mask])
                np.testing.assert_array_equal(prediction[mask], np.asarray(source)[mask])
                if projected:
                    path = out / f"predictions/partial-d6-{m}_projected.npz"
                    np.savez_compressed(path, physical=physical, prediction=prediction)
                    bindings[str(path)] = sha(path)
                else:
                    path = parent / f"evaluation/predictions/partial-d6-{m}.npz"
                    with np.load(path) as saved:
                        np.testing.assert_array_equal(physical, saved["physical"])
                        np.testing.assert_array_equal(prediction, saved["prediction"])
                    replay.append(str(path))
    names = (*METHODS, *[f"{m}_projected" for m in METHODS])
    checks = [
        dict(c, depth=3, draw=r["draw"])
        for r in rows
        if r["depth"] == 3 and r["method"] in names
        for c in fit_checks([r])
    ]
    guards = []
    for m in METHODS:
        for i in TRAIN + PROBE:
            for draw in range(4):
                values = {
                    r["method"]: r
                    for r in rows
                    if r["case"] == i and r["depth"] == 3 and r["draw"] == draw
                }
                for metric, limit, lower in (
                    ("ink_f1_alpha50", 0.005, False),
                    ("ink_deficit_fraction", 0.01, True),
                ):
                    a, b = values[f"{m}_projected"][metric], values[m][metric]
                    guards.append(
                        dict(
                            method=m,
                            case=i,
                            draw=draw,
                            metric=metric,
                            projected=a,
                            original=b,
                            passed=a <= b + limit if lower else a >= b - limit,
                        )
                    )
    assert len(rows) == 392 and len(bindings) == 102 and len(replay) == 52
    assert len(checks) == 352 and len(guards) == 128 and len(traces) == 8
    write(root / "replay-audit.json", dict(exact=True, files=replay))
    write(out / "trajectory.json", dict(root_level_only=True, traces=traces))
    write(
        out / "metrics.json",
        dict(
            rows=rows,
            checks=checks,
            retention_checks=guards,
            finite=True,
            protected_exact=True,
            bindings=bindings,
            adoption=False,
        ),
    )
    render(root, plan)
    write(root / "workflow-status.json", dict(stage="numerical_complete_pending_review", updates=0))


def render(root, plan):
    out = root / "evaluation"
    parent_out = Path(plan["parent"]) / "evaluation"
    records = []
    with np.load(root / "inputs.npz") as data:
        for method in METHODS:
            for split, ids in (("train", TRAIN), ("probe", PROBE)):
                for native in (True, False) if method == "interval" else (True,):
                    canvas = Image.new("L", (160 * 10, 184 * 4), 255)
                    predictions = []
                    for projected in (False, True):
                        label = f"{method}_projected" if projected else method
                        base = out if projected else parent_out
                        for draw in range(4):
                            with np.load(base / f"predictions/{split}-d3-r{draw}-{label}.npz") as a:
                                predictions.append(a["prediction"])
                    labels = (
                        "target",
                        "source",
                        *[f"{m}[{n}]" for m in ("original", "projected") for n in range(4)],
                    )
                    for row, i in enumerate(ids):
                        values = [
                            data[f"{i}-image"],
                            data[f"{i}-source"],
                            *[p[row] for p in predictions],
                        ]
                        for col, (label, value) in enumerate(zip(labels, values, strict=True)):
                            tile = Image.fromarray(
                                np.rint(np.clip((value[..., 0] + 1) * 127.5, 0, 255)).astype(
                                    np.uint8
                                )
                            )
                            tile = (
                                tile.crop(plan["crops"][str(i)])
                                if native
                                else tile.resize((160, 160), Image.Resampling.LANCZOS)
                            )
                            canvas.paste(tile, (col * 160, row * 184 + 24))
                            ImageDraw.Draw(canvas).text(
                                (col * 160 + 2, row * 184 + 4), f"{i} {label}", fill=0
                            )
                    path = out / f"{method}-{split}-{'native' if native else 'overview'}.png"
                    canvas.save(path)
                    records.append(dict(file=path.name, sha256=sha(path)))
        canvas = Image.new("L", (640, 736), 255)
        with np.load(parent_out / "predictions/partial-d6-interval.npz") as a:
            original = a["prediction"]
        with np.load(out / "predictions/partial-d6-interval_projected.npz") as a:
            projected = a["prediction"]
        for row, i in enumerate(TRAIN):
            target = np.where(
                np.arange(512)[None, :, None] >= 256, data[f"{i}-image"], data[f"{i}-source"]
            )
            for col, (label, value) in enumerate(
                zip(
                    ("masked_target", "source", "original", "projected"),
                    (target, data[f"{i}-source"], original[row], projected[row]),
                    strict=True,
                )
            ):
                tile = Image.fromarray(
                    np.rint(np.clip((value[..., 0] + 1) * 127.5, 0, 255)).astype(np.uint8)
                )
                canvas.paste(tile.crop(plan["crops"][str(i)]), (col * 160, row * 184 + 24))
                ImageDraw.Draw(canvas).text((col * 160 + 2, row * 184 + 4), f"{i} {label}", fill=0)
        path = out / "interval-partial-native.png"
        canvas.save(path)
        records.append(dict(file=path.name, sha256=sha(path)))
    assert len(records) == 7
    write(out / "render-manifest.json", dict(records=records, all_four_fixed_draws=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "freeze", "evaluate"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    globals()[args.action](args.root.resolve())
