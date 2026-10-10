"""Supplementary paired old/new corruption evaluation; never relaxes S's screen."""

import argparse
from collections import defaultdict
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from evaluate_core_retention_data import pixels
from flax import nnx
from PIL import Image, ImageDraw
from train_core_retention_data import check

from lineartgen.config import SketchAugmentationConfig
from lineartgen.evaluation import edit_boundary_metrics, probe_metrics, protected_metrics
from lineartgen.evaluation_sampling import background_metrics, inference_with_source, runtime_inputs
from lineartgen.lineart_metrics import halo_metrics
from lineartgen.models.serialization import load_inference
from lineartgen.sketch import augment_sketch

METHODS = ("initial", "pixel", "core50")
INPUTS = Path("runs/core-retention-transfer-20261010").resolve()
ROOT = Path("runs/core-retention-data-20261010").resolve()
PARENT = ROOT.parents[2] / "jax-source-interval-l1-20261010/runs/source-interval-l1-20261010"


def prepare():
    assert not INPUTS.exists()
    INPUTS.mkdir(parents=True)
    ref = read(PARENT / "reference.json")
    base = Path(ref["root"])
    assert sha(base / "cases.json") == ref["cases_sha256"]
    manifest = read(base / "cases.json")
    records = [(i, c) for i, c in enumerate(manifest["cases"]) if c.get("family") == "strokes"]
    assert len(records) == 376
    configs, operations = {}, {}
    for severity, ranges in manifest["corruption_ranges"].items():
        for blend in (1.0, 0.5):
            config = SketchAugmentationConfig(
                mode="strokes", probability=1, dilation_blend=blend, **ranges
            )
            configs[f"{severity}-{blend}"] = config.model_dump(mode="json")
            operations[severity, blend] = jax.jit(lambda x, k, c=config: augment_sketch(x, k, c))
    cache, rows, maximum = {}, [], 0.0
    for index, case in records:
        path = base / case["file"]
        assert sha(path) == ref["files"][case["file"]]
        with np.load(path) as a:
            data = {k: a[k] for k in a.files}
        key = (case["corruption_seed"], case["severity"])
        if key not in cache:
            clean = data["image"][None]
            rng = jax.random.PRNGKey(case["corruption_seed"])
            cache[key] = [
                np.asarray(operations[case["severity"], blend](clean, rng))[0]
                for blend in (1.0, 0.5)
            ]
        old, new = cache[key]
        difference = float(np.max(np.abs(old - data["source"])))
        assert difference <= 1e-6
        maximum = max(maximum, difference)
        data["old_source"] = data["source"]
        data["source"] = new
        output = INPUTS / case["file"]
        np.savez_compressed(output, **data)
        rows.append(
            dict(
                case,
                case=index,
                sha256=sha(output),
                legacy_sha256=sha(path),
                source_changed=not np.array_equal(new, data["old_source"]),
            )
        )
    panels = read(PARENT / "plan.json")
    groups = {
        name: ids
        for name, ids in panels["panel_groups"].items()
        if name in ("strokes-01", "strokes-05", "partial")
    }
    eligible = {i for i, _ in records}
    groups["real-strokes"] = [i for i in panels["panel_groups"]["real"] if i in eligible]
    groups = {k: [i for i in v if i in eligible] for k, v in groups.items()}
    write(
        INPUTS / "manifest.json",
        dict(
            cases=rows,
            configurations=configs,
            legacy_reference=ref,
            legacy_cases_sha256=ref["cases_sha256"],
            legacy_reproduction_max_error=maximum,
            legacy_reproduction_tolerance=1e-6,
            unique_corruptions=len(cache),
            panel_groups=groups,
            crops=panels["crops"],
            diagnostic_only=True,
            limitation=(
                "Same development teachers and corruption draws, dilation blend changed only. "
                "Strong corruption still partly exceeds training ranges."
            ),
        ),
    )
    print(
        dict(
            cases=len(rows),
            unique_corruptions=len(cache),
            changed=sum(r["source_changed"] for r in rows),
            legacy_max_error=maximum,
        ),
        flush=True,
    )


def evaluate(root):
    plan = check(root)
    assert jax.devices()[0].platform != "cpu"
    assert sha(INPUTS / "manifest.json") == plan["transfer_manifest_sha256"]
    manifest = read(INPUTS / "manifest.json")
    paths = dict(
        initial=Path(plan["initial"]),
        **{m: root / f"snapshots/step-4000/{m}/raw" for m in METHODS[1:]},
    )
    models = {m: load_inference(p)[0] for m, p in paths.items()}
    destination = root / "transfer"
    (destination / "predictions").mkdir(parents=True)
    binding = {
        str(p / n): sha(p / n)
        for p in paths.values()
        for n in ("config.json", "metadata.json", "weights.npz")
    }
    write(destination / "checkpoint-binding.json", dict(files=binding))
    forward = nnx.jit(lambda m, x, t, s: inference_with_source(m, x, t, s, steps=1))
    selected = set(sum(manifest["panel_groups"].values(), []))
    rows = []
    for ordinal, case in enumerate(manifest["cases"]):
        path = INPUTS / case["file"]
        assert sha(path) == case["sha256"]
        with np.load(path) as a:
            data = {k: a[k] for k in a.files}
        source, strength, clean = data["source"], data["strength"], data["image"]
        target = np.where(strength > 0, clean, source)
        inputs = jax.device_put(runtime_inputs(source[None], strength[None], case["seed"], 3))
        saved = {}
        for name, model in models.items():
            image = np.asarray(forward(model, *inputs, jax.device_put(source[None])))[0]
            assert np.isfinite(image).all()
            metric = protected_metrics(image, target, strength) | background_metrics(
                image, target, strength
            )
            metric |= edit_boundary_metrics(image, target, strength) | halo_metrics(
                image, target, strength
            )
            if "probes" in data and not case["partial"]:
                metric |= probe_metrics(image, target, data["probes"], data.get("connections"))
            assert metric["protected_max_error"] in (None, 0.0)
            assert all(v is None or np.isfinite(v) for v in metric.values())
            rows.append(
                dict(
                    case=case["case"],
                    method=name,
                    domain="real" if case["kind"] == "real" else "geometry",
                    kind=case["kind"],
                    family="strokes",
                    severity=case["severity"],
                    strength=case["strength"],
                    partial=case["partial"],
                    **metric,
                )
            )
            if case["case"] in selected:
                saved[name] = image
        if saved:
            np.savez_compressed(destination / f"predictions/case-{case['case']:04d}.npz", **saved)
        if (ordinal + 1) % 25 == 0 or ordinal + 1 == len(manifest["cases"]):
            write(
                destination / "progress.json", dict(cases=ordinal + 1, total=len(manifest["cases"]))
            )
    assert len(rows) == 1128
    assert {(x["case"], x["method"]) for x in rows} == {
        (x["case"], m) for x in manifest["cases"] for m in METHODS
    }
    write(
        destination / "metrics.json",
        dict(rows=rows, coverage_exact=True, protected_exact=True, diagnostic_only=True),
    )
    old = read(root / "development/metrics.json")["rows"]
    old = [x for x in old if x["family"] == "strokes"]
    assert len(old) == len(rows)
    groups = defaultdict(list)
    for variant, records in [("legacy", old), ("core50", rows)]:
        for record in records:
            groups[
                (
                    variant,
                    record["method"],
                    record["domain"],
                    record["severity"],
                    record["strength"],
                    record["partial"],
                )
            ].append(record)
    keys = [
        "ink_f1_alpha50",
        "ink_deficit_fraction",
        "halo_opacity_excess",
        "profile_width_mae_px",
        "connection_cut_rate",
        "connection_bridge_rate",
    ]
    summary = []
    for group, values in sorted(groups.items()):
        metrics = {
            k: float(np.mean([x[k] for x in values if x.get(k) is not None]))
            for k in keys
            if any(x.get(k) is not None for x in values)
        }
        summary.append(
            dict(
                zip(
                    ("input_variant", "method", "domain", "severity", "strength", "partial"),
                    group,
                    strict=True,
                ),
                cases=len(values),
                means=metrics,
            )
        )
    write(
        destination / "summary.json",
        dict(
            rows=summary,
            diagnostic_only=True,
            adoption=False,
            input_manifest_sha256=sha(INPUTS / "manifest.json"),
            legacy_metrics_sha256=sha(root / "development/metrics.json"),
        ),
    )
    review = destination / "native-review"
    review.mkdir()
    index = {x["case"]: x for x in manifest["cases"]}
    columns = ("target", "old_source", "core_source", *METHODS)
    for group, cases in manifest["panel_groups"].items():
        canvas = Image.new("L", (160 * len(columns), 184 * len(cases)), 255)
        draw = ImageDraw.Draw(canvas)
        for row, i in enumerate(cases):
            with np.load(INPUTS / index[i]["file"]) as a:
                data = {k: a[k] for k in a.files}
            values = dict(
                target=np.where(data["strength"] > 0, data["image"], data["source"]),
                old_source=data["old_source"],
                core_source=data["source"],
            )
            with np.load(destination / f"predictions/case-{i:04d}.npz") as a:
                values.update({m: a[m] for m in METHODS})
            for col, name in enumerate(columns):
                crop = Image.fromarray(pixels(values[name], name in METHODS)).crop(
                    manifest["crops"][str(i)]
                )
                canvas.paste(crop, (160 * col, 184 * row + 24))
                draw.text((160 * col + 2, 184 * row + 4), f"{i} {name}", fill=0)
        canvas.save(review / f"{group}.png")
    write(
        destination / "numerical-audit.json",
        dict(
            conditions=1128,
            methods=list(METHODS),
            unique_cases=376,
            protected_exact=True,
            coverage_exact=True,
            diagnostic_only=True,
            images=4,
            manifest_sha256=sha(INPUTS / "manifest.json"),
            metrics_sha256=sha(destination / "metrics.json"),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "evaluate"])
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    prepare() if args.action == "prepare" else evaluate(args.root.resolve())
