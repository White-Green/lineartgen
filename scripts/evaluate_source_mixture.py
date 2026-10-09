"""Frozen broader development evaluation of the synthetic mixture comparison."""

import argparse
from collections import defaultdict
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from flax import nnx
from lineartgen_runtime.images import pyramid_plan, to_bgra
from PIL import Image, ImageDraw
from train_source_mixture import check

from lineartgen.evaluation import edit_boundary_metrics, probe_metrics, protected_metrics
from lineartgen.evaluation_sampling import background_metrics, inference_with_source, runtime_inputs
from lineartgen.lineart_metrics import halo_metrics
from lineartgen.models.serialization import load_inference

METHODS = ("initial", "pixel", "mixed")


def load_case(plan, ref, index):
    if index >= plan["base_cases"]:
        j = index - plan["base_cases"]
        source = np.ones((512, 512, 1), np.float32)
        strength = np.full_like(source, (0.1, 0.5)[j % 2])
        if j >= 2:
            strength[:128] = 0
            strength[:, :170] = 0
        return dict(
            kind="blank",
            family="blank",
            severity="blank",
            strength=(0.1, 0.5)[j % 2],
            partial=j >= 2,
            seed=2026100953,
        ), dict(image=source, source=source, strength=strength)
    case = read(Path(ref["root"]) / "cases.json")["cases"][index]
    path = Path(ref["root"]) / case["file"]
    assert sha(path) == ref["files"][case["file"]]
    with np.load(path, allow_pickle=False) as data:
        return case, {k: data[k] for k in data.files}


def evaluate(root):
    plan = check(root)
    assert jax.devices()[0].platform != "cpu"
    assert read(root / "training/restore-audit-4000.json")["all_state_exact"]
    ref = read(root / "reference.json")
    assert sha(Path(ref["root"]) / "cases.json") == ref["cases_sha256"]
    paths = dict(
        initial=Path(plan["initial"]),
        **{m: root / f"snapshots/step-4000/{m}/raw" for m in METHODS[1:]},
    )
    models = {m: load_inference(p)[0] for m, p in paths.items()}
    out = root / "development"
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    write(
        out / "checkpoint-binding.json",
        {
            "files": {
                str(p / n): sha(p / n)
                for p in paths.values()
                for n in ("weights.npz", "metadata.json", "config.json")
            }
        },
    )
    forward = nnx.jit(lambda m, x, t, s: inference_with_source(m, x, t, s, steps=1))
    total = plan["base_cases"] + plan["blank_cases"]
    selected = set(sum(plan["panel_groups"].values(), [])) | set(range(plan["base_cases"], total))
    rows = []
    for index in range(total):
        case, data = load_case(plan, ref, index)
        clean, strength = data["image"], data["strength"]
        source = data.get("source", clean)
        target = np.where(strength > 0, clean, source)
        inputs = jax.device_put(runtime_inputs(source[None], strength[None], case["seed"], 3))
        source_device = jax.device_put(source[None])
        saved = {}
        for name, model in models.items():
            image = np.asarray(forward(model, *inputs, source_device))[0]
            assert np.isfinite(image).all()
            metric = protected_metrics(image, target, strength) | background_metrics(
                image, target, strength
            )
            metric |= edit_boundary_metrics(image, target, strength) | halo_metrics(
                image, target, strength
            )
            if "probes" in data and not case["partial"]:
                metric |= probe_metrics(image, target, data["probes"], data.get("connections"))
            if case["kind"] == "blank":
                alpha = (1 - image) / 2
                metric |= dict(
                    blank_max_opacity=float(alpha.max()), blank_mean_opacity=float(alpha.mean())
                )
            assert metric["protected_max_error"] in (None, 0.0)
            assert all(v is None or np.isfinite(v) for v in metric.values())
            rows.append(
                dict(
                    case=index,
                    method=name,
                    domain="real" if case["kind"] == "real" else "geometry",
                    kind=case["kind"],
                    family=case.get("family", "clean"),
                    severity=case.get("severity", "clean"),
                    strength=case["strength"],
                    partial=case["partial"],
                    **metric,
                )
            )
            if index in selected:
                saved[name] = image
        if index in selected:
            np.savez_compressed(out / f"predictions/case-{index:04d}.npz", **saved)
        if (index + 1) % 25 == 0 or index + 1 == total:
            write(out / "progress.json", dict(cases=index + 1, total=total, rows=len(rows)))
            print(f"evaluated {index + 1}/{total}; rows={len(rows)}", flush=True)
    assert len(rows) == total * len(METHODS)
    assert {(r["case"], r["method"]) for r in rows} == {
        (i, m) for i in range(total) for m in METHODS
    }
    write(
        out / "metrics.json",
        dict(
            rows=rows,
            conditions=len(rows),
            coverage_exact=True,
            protected_exact=True,
            source_sha256=sha(root / "source.json"),
            binding_sha256=sha(out / "checkpoint-binding.json"),
            adoption=False,
        ),
    )
    summarize(root)
    render(root)


def summarize(root):
    out = root / "development"
    rows = read(out / "metrics.json")["rows"]
    groups = defaultdict(list)
    keys = (
        "ink_f1_alpha50",
        "ink_deficit_fraction",
        "halo_opacity_excess",
        "profile_width_mae_px",
        "connection_cut_rate",
        "connection_bridge_rate",
        "blank_max_opacity",
        "blank_mean_opacity",
    )
    for r in rows:
        groups[tuple(r[k] for k in ("method", "domain", "family", "strength", "partial"))].append(r)
    means = {
        g: {
            k: float(np.mean([r[k] for r in v if r.get(k) is not None]))
            for k in keys
            if any(r.get(k) is not None for r in v)
        }
        for g, v in groups.items()
    }
    summaries = [
        dict(
            zip(("method", "domain", "family", "strength", "partial"), g, strict=True),
            cases=len(groups[g]),
            means=m,
        )
        for g, m in sorted(means.items())
    ]
    checks = []
    candidate = means["mixed", "real", "strokes", 0.5, False]
    control = means["pixel", "real", "strokes", 0.5, False]
    for name, passed in [
        ("strokes_f1_plus_002", candidate["ink_f1_alpha50"] >= control["ink_f1_alpha50"] + 0.02),
        (
            "strokes_halo_down_10percent",
            candidate["halo_opacity_excess"] <= control["halo_opacity_excess"] * 0.9,
        ),
    ]:
        checks.append(dict(guard=name, passed=passed))
    for group, candidate in means.items():
        method, domain, family, strength, partial = group
        if (
            method != "mixed"
            or family not in ("clean", "tone", "blur")
            or strength not in (0.1, 0.5)
        ):
            continue
        for ref in ("pixel", "initial"):
            control = means[(ref, *group[1:])]
            tolerances = [("ink_f1_alpha50", -0.005), ("ink_deficit_fraction", 0.01)]
            if domain == "geometry" and not partial:
                tolerances += [
                    ("profile_width_mae_px", 0.10),
                    ("connection_cut_rate", 0.01),
                    ("connection_bridge_rate", 0.01),
                ]
            for key, tolerance in tolerances:
                if key not in candidate or key not in control:
                    continue
                passed = (
                    candidate[key] >= control[key] + tolerance
                    if tolerance < 0
                    else candidate[key] <= control[key] + tolerance
                )
                checks.append(
                    dict(
                        guard=key,
                        reference=ref,
                        domain=domain,
                        family=family,
                        strength=strength,
                        partial=partial,
                        candidate=candidate[key],
                        control=control[key],
                        passed=passed,
                    )
                )
    blankmax = max(
        r["blank_max_opacity"] for r in rows if r["method"] == "mixed" and r["kind"] == "blank"
    )
    checks.append(dict(guard="blank_max_opacity_005", candidate=blankmax, passed=blankmax <= 0.05))
    write(
        out / "summary.json",
        dict(
            rows=summaries,
            checks=checks,
            failures=[r for r in checks if not r["passed"]],
            all_passed=all(r["passed"] for r in checks),
            adoption=False,
        ),
    )


def pixels(img, output=False):
    if output:
        return (
            255
            - np.frombuffer(to_bgra(img[None], pyramid_plan(512, 512)), np.uint8).reshape(
                512, 512, 4
            )[..., 3]
        )
    return np.rint(np.clip((img[..., 0] + 1) * 127.5, 0, 255)).astype(np.uint8)


def render(root):
    plan, ref = check(root), read(root / "reference.json")
    out = root / "development"
    destination = out / "native-review"
    destination.mkdir(exist_ok=True)
    records = []
    columns = ("target", "source", *METHODS)

    def sheet(name, items):
        canvas = Image.new("L", (160 * len(columns), 184 * len(items)), 255)
        bindings = []
        for row, (index, rect) in enumerate(items):
            _, data = load_case(plan, ref, index)
            source = data.get("source", data["image"])
            target = np.where(data["strength"] > 0, data["image"], source)
            p = out / f"predictions/case-{index:04d}.npz"
            with np.load(p) as predictions:
                for col, label in enumerate(columns):
                    img = (
                        target
                        if label == "target"
                        else source
                        if label == "source"
                        else predictions[label]
                    )
                    canvas.paste(
                        Image.fromarray(pixels(img, label not in ("target", "source"))).crop(rect),
                        (160 * col, 184 * row + 24),
                    )
                    ImageDraw.Draw(canvas).text(
                        (160 * col + 2, 184 * row + 4), f"{index} {label}", fill=0
                    )
            bindings.append(dict(case=index, crop=rect, predictions_sha256=sha(p)))
        path = destination / name
        canvas.save(path)
        records.append(dict(file=name, sha256=sha(path), inputs=bindings))

    for name, ids in plan["panel_groups"].items():
        sheet(f"{name}.png", [(i, plan["crops"][str(i)]) for i in ids])
    worst = []
    for method in METHODS:
        choices = []
        for i in range(plan["base_cases"], plan["base_cases"] + plan["blank_cases"]):
            with np.load(out / f"predictions/case-{i:04d}.npz") as p:
                alpha = (1 - p[method][..., 0]) / 2
                y, x = np.unravel_index(np.argmax(alpha), alpha.shape)
                choices.append((float(alpha[y, x]), i, int(x), int(y)))
        value, index, x, y = max(choices)
        x0, y0 = int(np.clip(x - 80, 0, 352)), int(np.clip(y - 80, 0, 352))
        worst.append(
            dict(
                method=method,
                case=index,
                max_opacity=value,
                x=x,
                y=y,
                crop=[x0, y0, x0 + 160, y0 + 160],
            )
        )
    sheet("blank-worst.png", [(r["case"], r["crop"]) for r in worst])
    write(
        destination / "manifest.json",
        dict(
            records=records,
            blank_worst=worst,
            display="Native Krita-white pixels; float input, not actual BGRA API",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["evaluate", "summarize", "render"])
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    globals()[args.action](args.root.resolve())
