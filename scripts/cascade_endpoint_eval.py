"""Compare recursive/cascaded sampling and a clean-small-insert oracle."""

import argparse
import shutil
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from cascade_endpoint import sample_cascade
from diagnose_learnability import freeze, verify
from endpoint_time_grid import velocity
from geometry_endpoint_fit import arrays
from paper_band_fit import PROBE, TRAIN, fit_checks
from PIL import Image, ImageDraw
from velocity_bridge_toy import sample

from lineartgen.coverage_metrics import coverage_metrics
from lineartgen.evaluation import probe_metrics, protected_metrics
from lineartgen.lineart_metrics import halo_metrics
from lineartgen.models.serialization import load_inference

ANCHORS = ("control", "band")
SCHEMES = ("recursive", "cascade", "oracle")
METHODS = tuple(f"{a}_{n}" for a in ANCHORS for n in SCHEMES)
EVAL_SEED = 2026102093


def prepare(root):
    assert not root.exists()
    parent = Path(
        "../jax-source-band-endpoint-20261011/runs/source-band-endpoint-20261011"
    ).resolve()
    from source_band_fit import check as parent_check

    parent_check(parent)
    assert read(parent / "final-audit.json")["stage"] == "complete_source_band_endpoint_diagnostic"
    (root / "configs").mkdir(parents=True)
    for a in ANCHORS:
        shutil.copytree(parent / f"snapshots/step-7200/{a}/raw", root / "models" / a)
        shutil.copyfile(parent / f"configs/{a}.json", root / f"configs/{a}.json")
    for n in ("inputs.npz", "probe.json"):
        shutil.copyfile(parent / n, root / n)
    write(root / "train.json", dict(neural_updates=0, checkpoint=7200))
    bound = [parent / n for n in ("source.json", "final-audit.json", "evaluation/metrics.json")]
    bound += [p for a in ANCHORS for p in (parent / f"snapshots/step-7200/{a}/raw").iterdir()]
    bound += [p for a in ANCHORS for p in (parent / "evaluation/predictions").glob(f"*-{a}.npz")]
    initial = root / "models"
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
            bound_files={str(p): sha(p) for p in bound},
            crops=read(parent / "plan.json")["crops"],
            anchors=list(ANCHORS),
            schemes=list(SCHEMES),
            adoption=False,
            plan_document_sha256=sha(Path("docs/cascade-endpoint-plan-20261011.md")),
        ),
    )


def check(root):
    verify(root)
    plan = read(root / "plan.json")
    from source_band_fit import check as parent_check

    parent_check(Path(plan["parent"]))
    for p, h in plan["bound_files"].items():
        assert sha(Path(p)) == h
    assert sha(Path("docs/cascade-endpoint-plan-20261011.md")) == plan["plan_document_sha256"]
    return plan


def run_sample(model, source, strength, key, depth, method, target=None):
    assert method in METHODS
    mode = method.split("_")[1]
    if mode == "recursive":
        if target is not None:
            raise ValueError("Recursive sampling never accepts a teacher")
        return np.asarray(
            sample(
                lambda x, t, s, u: velocity(model, x, t, s, u, False),
                source,
                strength,
                key,
                depth=depth,
                steps=32,
            )
        )
    return np.asarray(
        sample_cascade(
            model,
            source,
            strength,
            key,
            depth=depth,
            steps=32,
            target=target,
            oracle=mode == "oracle",
        )
    )


def evaluate(root):
    plan = check(root)
    assert read(root / "test-results.json")["passed"] and jax.devices()[0].platform != "cpu"
    anchors = {a: load_inference(root / "models" / a)[0] for a in ANCHORS}
    models = {m: anchors[m.split("_")[0]] for m in METHODS}
    out = root / "evaluation"
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    old_metrics = read(Path(plan["parent"]) / "evaluation/metrics.json")
    old_rows = {
        (r["method"], r["case"], r["depth"], r["draw"]): r
        for r in old_metrics["rows"]
        if r["method"] in ANCHORS
    }
    rows = []
    bindings = {}
    replay = []
    displays = {}
    partial_displays = {}
    with np.load(root / "inputs.npz") as data:
        for split, ids in (("train", TRAIN), ("probe", PROBE)):
            clean, source, strength = arrays(root, ids)
            for depth in (1, 3, 6):
                for draw in range(4):
                    key = jax.random.fold_in(
                        jax.random.PRNGKey(EVAL_SEED),
                        depth * 100 + int(split == "probe") * 10 + draw,
                    )
                    active_models = dict(models)
                    for method, model in active_models.items():
                        physical = run_sample(
                            model,
                            source,
                            strength,
                            key,
                            depth,
                            method,
                            clean if method.endswith("_oracle") else None,
                        )
                        prediction = np.clip(physical, -1, 1)
                        assert np.isfinite(physical).all()
                        if method.endswith("_recursive"):
                            anchor = method.split("_")[0]
                            reference = (
                                Path(plan["parent"])
                                / f"evaluation/predictions/{split}-d{depth}-r{draw}-{anchor}.npz"
                            )
                            with np.load(reference) as a:
                                np.testing.assert_array_equal(physical, a["physical"])
                                np.testing.assert_array_equal(prediction, a["prediction"])
                            replay.append(str(reference))
                        path = out / f"predictions/{split}-d{depth}-r{draw}-{method}.npz"
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
                                    method=method,
                                    case=i,
                                    split=split,
                                    depth=depth,
                                    draw=draw,
                                    **metrics,
                                )
                            )
                            if method.endswith("_recursive"):
                                assert {**rows[-1], "method": method.split("_")[0]} == old_rows[
                                    method.split("_")[0], i, depth, draw
                                ]
                            if depth == 3 and method in METHODS:
                                displays[(method, i, draw)] = value
                    write(
                        out / "progress.json",
                        dict(split=split, depth=depth, draw=draw, rows=len(rows), total=576),
                    )
        clean, source, strength = arrays(root, TRAIN)
        strength = strength.at[:, :, :256].set(0)
        key = jax.random.fold_in(jax.random.PRNGKey(EVAL_SEED), 9000)
        for method, model in models.items():
            physical = run_sample(
                model,
                source,
                strength,
                key,
                6,
                method,
                clean if method.endswith("_oracle") else None,
            )
            prediction = np.clip(physical, -1, 1)
            mask = np.asarray(strength) == 0
            assert np.isfinite(physical).all()
            np.testing.assert_array_equal(physical[mask], np.asarray(source)[mask])
            np.testing.assert_array_equal(prediction[mask], np.asarray(source)[mask])
            if method.endswith("_recursive"):
                reference = (
                    Path(plan["parent"])
                    / f"evaluation/predictions/partial-d6-{method.split('_')[0]}.npz"
                )
                with np.load(reference) as a:
                    np.testing.assert_array_equal(physical, a["physical"])
                    np.testing.assert_array_equal(prediction, a["prediction"])
                replay.append(str(reference))
            path = out / f"predictions/partial-d6-{method}.npz"
            np.savez_compressed(path, physical=physical, prediction=prediction)
            bindings[str(path)] = sha(path)
            for i, value in zip(TRAIN, prediction, strict=True):
                partial_displays[(method, i)] = value
        assert len(replay) == 50
        write(root / "replay-audit.json", dict(exact=True, metric_rows_exact=192, files=replay))
        checks = []
        for row in rows:
            if row["depth"] == 3 and row["method"] in METHODS:
                for c in fit_checks([row]):
                    checks.append(dict(c, depth=3, draw=row["draw"]))
        guards = []
        for i in TRAIN + PROBE:
            for draw in range(4):
                values = {
                    r["method"]: r
                    for r in rows
                    if r["case"] == i and r["depth"] == 3 and r["draw"] == draw
                }
                for compared, reference in (
                    (f"{a}_{n}", f"{a}_recursive") for a in ANCHORS for n in ("cascade", "oracle")
                ):
                    for metric, limit, lower in (
                        ("ink_f1_alpha50", 0.005, False),
                        ("ink_deficit_fraction", 0.01, True),
                    ):
                        a, b = values[compared][metric], values[reference][metric]
                        guards.append(
                            dict(
                                case=i,
                                draw=draw,
                                metric=metric,
                                method=compared,
                                reference=reference,
                                passed=a <= b + limit if lower else a >= b - limit,
                                value=a,
                                reference_value=b,
                            )
                        )
        assert (
            len(rows) == 576 and len(checks) == 528 and len(guards) == 256 and len(bindings) == 150
        )
        write(
            out / "metrics.json",
            dict(
                rows=rows,
                checks=checks,
                retention_checks=guards,
                fit_passed={
                    m: all(c["passed"] for c in checks if c["method"] == m) for m in METHODS
                },
                retention_passed=all(c["passed"] for c in guards),
                finite=True,
                protected_exact=True,
                bindings=bindings,
                adoption=False,
            ),
        )
        records = []
        for anchor in ANCHORS:
            methods = [f"{anchor}_{n}" for n in SCHEMES]
            columns = ("target", "source", *[f"{n}[{d}]" for n in SCHEMES for d in range(4)])
            for split, ids in (("train", TRAIN), ("probe", PROBE)):
                for native in (True, False):
                    canvas = Image.new("L", (160 * len(columns), 184 * len(ids)), 255)
                    for row, i in enumerate(ids):
                        values = [
                            data[f"{i}-image"],
                            data[f"{i}-source"],
                            *[displays[m, i, d] for m in methods for d in range(4)],
                        ]
                        for col, (label, value) in enumerate(zip(columns, values, strict=True)):
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
                            canvas.paste(tile, (160 * col, 184 * row + 24))
                            ImageDraw.Draw(canvas).text(
                                (160 * col + 2, 184 * row + 4), f"{i} {label}", fill=0
                            )
                    path = out / f"{anchor}-{split}-{'native' if native else 'overview'}.png"
                    canvas.save(path)
                    records.append(dict(file=path.name, sha256=sha(path)))
            canvas = Image.new("L", (160 * (2 + len(SCHEMES)), 184 * 4), 255)
            for row, i in enumerate(TRAIN):
                target = np.where(
                    np.asarray(strength[row]) > 0, data[f"{i}-image"], data[f"{i}-source"]
                )
                values = [target, data[f"{i}-source"], *[partial_displays[m, i] for m in methods]]
                for col, (label, value) in enumerate(
                    zip(
                        ("masked_target", "source", *[f"{n}" for n in SCHEMES]), values, strict=True
                    )
                ):
                    tile = Image.fromarray(
                        np.rint(np.clip((value[..., 0] + 1) * 127.5, 0, 255)).astype(np.uint8)
                    ).crop(plan["crops"][str(i)])
                    canvas.paste(tile, (160 * col, 184 * row + 24))
                    ImageDraw.Draw(canvas).text(
                        (160 * col + 2, 184 * row + 4), f"{i} {label}", fill=0
                    )
            path = out / f"{anchor}-partial-native.png"
            canvas.save(path)
            records.append(dict(file=path.name, sha256=sha(path)))
        write(
            out / "render-manifest.json",
            dict(
                records=records,
                all_four_fixed_draws=True,
                partial_one_draw_per_case=True,
                crops_from_AD=True,
            ),
        )
    write(
        root / "workflow-status.json",
        dict(stage="numerical_complete_pending_review", neural_updates=0),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "freeze", "evaluate"))
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    {"prepare": prepare, "freeze": freeze, "evaluate": evaluate}[args.action](args.root.resolve())
