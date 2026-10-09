"""Frozen whole/partial clean, tone and blur evaluation for online geometry."""

import argparse
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from diagnose_learnability import SEED as EVALUATION_SEED
from diagnose_learnability import fixed_inputs, verify
from fit_single_cases import forward
from PIL import Image, ImageDraw
from train_procedural_structure import METHODS, STEPS

from lineartgen.diffusion import GuidedTrainingInputs, pyramid
from lineartgen.lineart_metrics import halo_metrics
from lineartgen.models.serialization import load_inference


def evaluate(root, step):
    methods = read(root / "plan.json").get("comparison_methods", list(METHODS))
    verify(root)
    plan = read(root / "plan.json")
    out = root / f"broad/step-{step}"
    out.mkdir(parents=True, exist_ok=True)
    rows, bindings = [], {}
    for split in ("train", "probe"):
        data = np.load(plan["data"][split]["path"])
        records = read(root / f"{split}.json")
        cached, info = [], []
        for partial in (False, True):
            for strength in (0.1, 0.5):
                for i, record in enumerate(records):
                    clean, source = (data[k][i : i + 1] for k in ("clean", "source"))
                    noisy, noise, guide, target, _ = fixed_inputs(
                        clean,
                        source,
                        strength,
                        EVALUATION_SEED + 10000 + i + (1000 if split == "probe" else 0),
                        partial=partial,
                    )
                    cached.append(
                        GuidedTrainingInputs(
                            pyramid(jax.numpy.asarray(target), 3), noise, noisy, guide
                        )
                    )
                    info.append(dict(split=split, partial=partial, strength=strength, **record))
        for method in methods:
            folder = root / f"snapshots/step-{step}/{method}/raw"
            model, _ = load_inference(folder)
            bindings.update(
                {
                    str(folder / n): sha(folder / n)
                    for n in ("weights.npz", "metadata.json", "config.json")
                }
            )
            arrays = []
            for start in range(0, len(cached), 4):
                selected = cached[start : start + 4]
                batch = jax.tree.map(lambda *x: jax.numpy.concatenate(x), *selected)
                predicted = np.asarray(forward(model, batch))
                for local, image in enumerate(predicted):
                    p = selected[local]
                    target, noise = (
                        np.asarray(getattr(p, key)[0])[0] for key in ("targets", "noise_levels")
                    )
                    metrics = halo_metrics(image, target, noise)
                    assert np.isfinite(image).all() and metrics["protected_max_error"] in (
                        None,
                        0.0,
                    )
                    rows.append(dict(method=method, **info[start + local], **metrics))
                    arrays.append(image)
            np.savez_compressed(out / f"{method}-{split}.npz", images=np.stack(arrays))
            print(dict(step=step, method=method, split=split, conditions=len(arrays)), flush=True)
    write(
        out / "metrics.json",
        dict(
            rows=rows,
            step=step,
            bindings=bindings,
            plan_sha256=sha(root / "plan.json"),
            script_sha256=sha(Path(__file__)),
            adoption=False,
        ),
    )
    summarize(root, step)
    render(root, step)


def summarize(root, step):
    methods = read(root / "plan.json").get("comparison_methods", list(METHODS))
    out = root / f"broad/step-{step}"
    rows = read(out / "metrics.json")["rows"]
    summaries, failures = [], []
    for split in ("train", "probe"):
        for family in ("clean", "tone", "blur"):
            for partial in (False, True):
                for strength in (0.1, 0.5):
                    selected = [
                        r
                        for r in rows
                        if r["split"] == split
                        and r["family"] == family
                        and r["partial"] == partial
                        and r["strength"] == strength
                        and r["kind"] != "blank"
                    ]
                    means = {}
                    for method in methods:
                        group = [r for r in selected if r["method"] == method]
                        means[method] = {
                            k: float(np.mean([r[k] for r in group if r[k] is not None]))
                            for k in (
                                "ink_f1_alpha50",
                                "ink_deficit_fraction",
                                "halo_opacity_excess",
                            )
                        }
                    checks = {}
                    for method in (m for m in methods if m != "pixel"):
                        control, candidate = means["pixel"], means[method]
                        check = dict(
                            ink=candidate["ink_deficit_fraction"]
                            <= control["ink_deficit_fraction"] + 0.005,
                            f1=candidate["ink_f1_alpha50"] >= control["ink_f1_alpha50"] - 0.005,
                        )
                        checks[method] = check
                        failures.extend(
                            dict(
                                method=method,
                                split=split,
                                family=family,
                                partial=partial,
                                strength=strength,
                                guard=k,
                            )
                            for k, passed in check.items()
                            if not passed
                        )
                    summaries.append(
                        dict(
                            split=split,
                            family=family,
                            partial=partial,
                            strength=strength,
                            means=means,
                            checks=checks,
                        )
                    )
    halo_checks = []
    path_checks = []
    primary = read(root / f"evaluation/step-{step}/metrics.json")["rows"]
    for split in ("train", "probe"):
        means = {
            method: float(
                np.mean(
                    [
                        r["halo_opacity_excess"]
                        for r in rows
                        if r["method"] == method
                        and r["split"] == split
                        and r["family"] == "blur"
                        and r["kind"] != "blank"
                        and r["halo_opacity_excess"] is not None
                    ]
                )
            )
            for method in methods
        }
        for method in (m for m in methods if m != "pixel"):
            passed = means[method] <= 0.9 * means["pixel"]
            halo_checks.append(dict(split=split, method=method, means=means, passed=passed))
            if not passed:
                failures.append(dict(method=method, split=split, guard="blur_halo_10percent"))
        for case in ("fit_straight", "fit_thin", "fit_corner", "fit_gap"):
            means = {}
            keys = (
                "profile_width_mae_px",
                "connection_cut_rate",
                "connection_bridge_rate",
                "gap_middle_alpha",
            )
            for method in methods:
                group = [
                    r
                    for r in primary
                    if r["method"] == method and r["split"] == split and r["case"] == case
                ]
                assert len(group) == 8
                means[method] = {
                    k: float(np.mean([r[k] for r in group if r[k] is not None]))
                    for k in keys
                    if any(r[k] is not None for r in group)
                }
            for method in (m for m in methods if m != "pixel"):
                checks = {
                    k: means[method][k] <= means["pixel"][k] + tolerance
                    for k, tolerance in (
                        ("profile_width_mae_px", 0.10),
                        ("connection_cut_rate", 0.01),
                        ("connection_bridge_rate", 0.01),
                    )
                    if k in means[method]
                }
                if case == "fit_gap" and method == "structure" and "region" in methods:
                    checks["gap_reduction"] = (
                        means[method]["gap_middle_alpha"]
                        <= 0.75 * means["region"]["gap_middle_alpha"]
                    )
                path_checks.append(
                    dict(split=split, case=case, method=method, means=means, checks=checks)
                )
                failures.extend(
                    dict(method=method, split=split, case=case, guard=k)
                    for k, passed in checks.items()
                    if not passed
                )
    write(
        out / "summary.json",
        dict(
            summaries=summaries,
            halo_checks=halo_checks,
            path_checks=path_checks,
            failures=failures,
            metrics_sha256=sha(out / "metrics.json"),
            adoption=False,
        ),
    )


def render(root, step):
    methods = read(root / "plan.json").get("comparison_methods", list(METHODS))
    out = root / f"broad/step-{step}"
    plan = read(root / "plan.json")
    files = {}
    for split in ("train", "probe"):
        data = np.load(plan["data"][split]["path"])
        rows = read(root / f"{split}.json")
        results = {method: np.load(out / f"{method}-{split}.npz")["images"] for method in methods}
        # Native comparisons of every fixed first-variant shape. No interpolation.
        for strength, offset in ((0.1, 0), (0.5, 57)):
            canvas = Image.new("RGB", (160 * (2 + len(methods)), 180 * 9), "white")
            painter, crops = ImageDraw.Draw(canvas), []
            for row, pair in enumerate(range(38, 47)):
                target = data["clean"][pair]
                yy, xx = np.where(target[..., 0] < 0.96)
                y, x = (int(np.clip(np.mean(v) - 80, 0, 352)) for v in (yy, xx))
                values = [target, data["source"][pair]] + [
                    results[m][offset + pair] for m in methods
                ]
                for col, (name, image) in enumerate(
                    zip(("target", "source", *methods), values, strict=True)
                ):
                    gray = np.uint8(np.rint(np.clip((image[..., 0] + 1) * 127.5, 0, 255)))
                    canvas.paste(
                        Image.fromarray(gray[y : y + 160, x : x + 160]).convert("RGB"),
                        (160 * col, 180 * row + 20),
                    )
                    painter.text(
                        (160 * col + 2, 180 * row + 2), f"{rows[pair]['kind']} {name}", fill="black"
                    )
                crops.append(dict(pair=pair, crop=[x, y, x + 160, y + 160]))
            path = out / f"{split}-blur-{strength}-native.png"
            canvas.save(path)
            files[path.name] = dict(sha256=sha(path), crops=crops, inspected=False)
        # Full512px blank and protected-boundary diagnostics for one declared case.
        for label, index in (("blank", 57 + 56), ("partial_thin", 114 + 57 + 39)):
            canvas = Image.new("RGB", (512 * (2 + len(methods)), 536), "white")
            painter = ImageDraw.Draw(canvas)
            pair = index % 57
            values = [data["clean"][pair], data["source"][pair]] + [
                results[m][index] for m in methods
            ]
            for col, (name, image) in enumerate(
                zip(("clean target", "source", *methods), values, strict=True)
            ):
                gray = np.uint8(np.rint(np.clip((image[..., 0] + 1) * 127.5, 0, 255)))
                canvas.paste(Image.fromarray(gray).convert("RGB"), (512 * col, 24))
                painter.text((512 * col + 3, 3), name, fill="black")
            path = out / f"{split}-{label}-full.png"
            canvas.save(path)
            files[path.name] = dict(sha256=sha(path), pair=pair, index=index, inspected=False)
    write(
        out / "visual-manifest.json", dict(files=files, inspection_complete=False, adoption=False)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--step", type=int, default=STEPS)
    args = parser.parse_args()
    evaluate(args.root.resolve(), args.step)
