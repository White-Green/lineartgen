"""Fixed operational inputs for the nested pilot, separate from OT validation loss.

Prepare cases before inspecting trained outputs. Real cases cover every scale,
strengths 0.1/0.3/0.5/0.7, full/partial masks and identity. Synthetic curves,
junctions, intentional gaps and tapering lines supplement the real crops.
The final-test split is never read here.
"""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from flax import nnx
from PIL import Image, ImageDraw
from scipy import ndimage

from lineartgen.config import load_config
from lineartgen.data.images import read_crop
from lineartgen.data.pipeline import dataset_description, discover, split_experiment
from lineartgen.diffusion import sample
from lineartgen.training.checkpoints import load_run_model


def synthetic(kind, size=512):
    scale = 4
    image = Image.new("L", (size * scale, size * scale), 255)
    draw = ImageDraw.Draw(image)

    def line(points, width):
        draw.line(
            [(float(x) * scale, float(y) * scale) for x, y in points], fill=0, width=width * scale
        )

    if kind == "curve":
        t = np.linspace(0, 1, 1000)
        for offset, width in [(0, 2), (65, 4), (135, 7)]:
            line(zip(32 + 448 * t, 140 + offset + 65 * np.sin(2 * np.pi * t), strict=True), width)
    elif kind == "junction":
        line([(50, 90), (250, 280), (450, 80)], 3)
        line([(250, 280), (250, 455)], 3)
        line([(50, 390), (460, 390)], 5)
        line([(50, 410), (200, 410)], 2)
    elif kind == "gaps":
        for y, gap, width in [(90, 5, 2), (180, 10, 3), (280, 18, 5), (390, 30, 7)]:
            line([(32, y), (256 - gap / 2, y)], width)
            line([(256 + gap / 2, y), (480, y)], width)
    elif kind == "taper":
        for offset in [0, 120, 240]:
            t = np.linspace(0, 1, 600)
            x = 40 + 430 * t
            y = 100 + offset + 35 * np.sin(np.pi * t)
            radius = 0.5 + 5 * np.sin(np.pi * t)
            outline = list(zip(x, y - radius, strict=True)) + list(
                zip(x[::-1], (y + radius)[::-1], strict=True)
            )
            draw.polygon([(float(a) * scale, float(b) * scale) for a, b in outline], fill=0)
    else:
        raise ValueError(kind)
    return (
        np.asarray(image.resize((size, size), Image.Resampling.LANCZOS), np.float32)[..., None]
        / 255
    )


def prepare_cases(config, output):
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "cases.json"
    sources = discover(config.data)
    training, development, final = split_experiment(sources, config.data, config.train.seed)
    description = dataset_description(sources, training, development, final)
    data_settings = config.data.model_dump(
        mode="json", exclude={"workers", "prefetch_batches", "prepared_root"}
    )
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        previous = {
            k: v
            for k, v in existing["data_config"].items()
            if k not in {"workers", "prefetch_batches", "prepared_root"}
        }
        if existing["dataset"] != description or previous != data_settings:
            raise ValueError(
                "Evaluation inputs were prepared with a different dataset or data config"
            )
        return existing
    random = np.random.default_rng(20260910)
    size = config.data.crop_size
    records = []

    def store(gray, strength, partial, record):
        index = len(records)
        image = np.clip((gray * 2 - 1) * config.data.image_scale, -1, 1).astype(np.float32)
        noise = np.full_like(image, strength)
        if partial:
            noise[: size // 5] = 0
            noise[:, : size // 3] = 0
            noise[size * 3 // 4 :, size * 3 // 4 :] = 0
        filename = f"case-{index:03d}.npz"
        np.savez_compressed(output / filename, image=image, strength=noise)
        records.append(
            dict(record, file=filename, strength=strength, partial=partial, seed=20260910 + index)
        )

    for variant, scale_level in enumerate(config.data.resolved_scale_levels):
        for index in range(9):
            source = development[int(random.integers(len(development)))]
            path, width, height = source.variants[variant]
            x, y = int(random.integers(width - size + 1)), int(random.integers(height - size + 1))
            gray = read_crop(path, x, y, size, alpha_mode=config.data.alpha_mode)
            strength = [0.1, 0.3, 0.5, 0.7][index % 4] if index < 8 else 0.0
            store(
                gray,
                strength,
                4 <= index < 8,
                dict(kind="real", source=source.name, crop=[x, y, size], scale_level=scale_level),
            )
    if size != 512:
        raise ValueError("Synthetic evaluation currently requires crop_size=512")
    for kind in ("curve", "junction", "gaps", "taper"):
        gray = synthetic(kind)
        for index, strength in enumerate([0.1, 0.3, 0.5, 0.7]):
            store(gray, strength, index >= 2, dict(kind=kind, scale_level=0))
    manifest = {
        "format_version": 1,
        "dataset": description,
        "data_config": data_settings,
        "cases": records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def geometry_metrics(prediction, target, strength):
    p, t = prediction[..., 0] < 0, target[..., 0] < 0
    intersection = np.count_nonzero(p & t)
    total = np.count_nonzero(p) + np.count_nonzero(t)
    p_edge = p & ~ndimage.binary_erosion(p)
    t_edge = t & ~ndimage.binary_erosion(t)
    if not p_edge.any() or not t_edge.any():
        boundary_f1 = float(not p_edge.any() and not t_edge.any())
    else:
        precision = float((ndimage.distance_transform_edt(~t_edge)[p_edge] <= 1.5).mean())
        recall = float((ndimage.distance_transform_edt(~p_edge)[t_edge] <= 1.5).mean())
        boundary_f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    protected = strength == 0
    components_p = ndimage.label(p, np.ones((3, 3)))[1]
    components_t = ndimage.label(t, np.ones((3, 3)))[1]
    return {
        "mae": float(np.abs(prediction - target).mean()),
        "ink_f1": 2 * intersection / total if total else 1.0,
        "boundary_f1_1_5px": boundary_f1,
        "ink_area_error": abs(int(p.sum()) - int(t.sum())) / max(int(t.sum()), 1),
        "component_count_error": abs(components_p - components_t),
        "target_components": components_t,
        "predicted_components": components_p,
        "target_ink_fraction": float(t.mean()),
        "protected_mae": float(np.abs(prediction - target)[protected].mean())
        if protected.any()
        else None,
    }


def summarize(results):
    """Keep blank crops and identity separate from the main line-quality comparison."""
    subsets = {
        "real": lambda r: r["kind"] == "real" and r["strength"] > 0,
        "real_with_lines": lambda r: (
            r["kind"] == "real" and r["strength"] > 0 and r["target_ink_fraction"] >= 0.001
        ),
        "real_blank": lambda r: (
            r["kind"] == "real" and r["strength"] > 0 and r["target_ink_fraction"] < 0.001
        ),
        "identity": lambda r: r["strength"] == 0,
        "synthetic": lambda r: r["kind"] != "real",
    }
    summaries = []
    for subset, select in subsets.items():
        for steps in sorted({r["denoise_steps"] for r in results}):
            for model in sorted({r["model"] for r in results}):
                rows = [
                    r
                    for r in results
                    if select(r) and r["denoise_steps"] == steps and r["model"] == model
                ]
                if not rows:
                    continue
                summary = dict(subset=subset, denoise_steps=steps, model=model, cases=len(rows))
                for metric in [
                    "mae",
                    "ink_f1",
                    "boundary_f1_1_5px",
                    "ink_area_error",
                    "component_count_error",
                    "protected_mae",
                ]:
                    values = [r[metric] for r in rows if r[metric] is not None]
                    summary[metric] = float(np.mean(values)) if values else None
                summaries.append(summary)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/nested.toml")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run-group", type=Path)
    parser.add_argument("--step", type=int)
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 5])
    parser.add_argument("--raw", action="store_true", help="Use raw weights instead of EMA")
    parser.add_argument("--limit", type=int, help="Limit cases for an evaluation preflight")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    config = load_config(args.config)
    manifest = prepare_cases(config, args.cases)
    if args.prepare_only:
        print(f"Prepared {len(manifest['cases'])} fixed cases: {args.cases}")
        return
    if args.run_group is None:
        parser.error("--run-group is required for evaluation")
    models = {}
    shared, saved_config, checkpoint_step = load_run_model(
        args.run_group / "shared", step=args.step, use_ema=not args.raw
    )
    if saved_config != config:
        raise ValueError("Evaluation config must match the shared run")
    models.update((f"shared_{i}", shared.extract(i)) for i in range(shared.num_subnets))
    core, core_config, _ = load_run_model(
        args.run_group / "independent_core", step=checkpoint_step, use_ema=not args.raw
    )
    if core_config.data != config.data or core_config.diffusion != config.diffusion:
        raise ValueError("Independent core uses different data or diffusion settings")
    models["independent_core"] = core
    if (args.run_group / "baseline" / "config.json").is_file():
        baseline, _, _ = load_run_model(
            args.run_group / "baseline", step=checkpoint_step, use_ema=not args.raw
        )
        models["baseline"] = baseline
    output = args.cases / f"step-{checkpoint_step}-{'raw' if args.raw else 'ema'}"
    output.mkdir(exist_ok=True)
    run_sample = nnx.jit(
        lambda model, image, strength, key, steps, levels: sample(
            model, image, key, strength=strength, steps=steps, levels=levels
        ),
        static_argnums=(4, 5),
    )
    results = []
    selected_cases = manifest["cases"][: args.limit]
    for index, case in enumerate(selected_cases):
        with np.load(args.cases / case["file"], allow_pickle=False) as arrays:
            target, strength = arrays["image"], arrays["strength"]
        image, noise = jax.device_put((target[None], strength[None]))
        levels = config.diffusion.levels_for_scale(
            config.data.crop_size, config.model.minimum_size, case["scale_level"]
        )
        key = jax.random.PRNGKey(case["seed"])
        display_key = key if levels == 1 else jax.random.split(key, levels)[0]
        noisy_input = np.asarray(
            image * jax.numpy.sqrt(1 - noise)
            + jax.random.normal(display_key, image.shape) * jax.numpy.sqrt(noise)
        )[0]
        for steps in args.steps:
            predictions = {}
            for name, model in models.items():
                prediction = np.asarray(run_sample(model, image, noise, key, steps, levels))[0]
                predictions[name] = prediction
                results.append(
                    dict(
                        case=index,
                        kind=case["kind"],
                        scale_level=case["scale_level"],
                        strength=case["strength"],
                        partial=case["partial"],
                        denoise_steps=steps,
                        model=name,
                        **geometry_metrics(prediction, target, strength),
                    )
                )
            # Save a fixed representative panel at every input condition.
            panel = np.concatenate([target, noisy_input, *predictions.values()], axis=1)[..., 0]
            pixels = np.rint(np.clip((panel + 1) * 127.5, 0, 255)).astype(np.uint8)
            canvas = Image.new("L", (pixels.shape[1], pixels.shape[0] + 24), 255)
            canvas.paste(Image.fromarray(pixels), (0, 24))
            draw = ImageDraw.Draw(canvas)
            for column, label in enumerate(["target", "noisy input", *models]):
                draw.text((column * target.shape[1] + 8, 5), label, fill=0)
            canvas.save(output / f"case-{index:03d}-steps-{steps}.png")
        print(f"Evaluated case {index + 1}/{len(selected_cases)}", flush=True)
    report = {
        "checkpoint_step": checkpoint_step,
        "ema": not args.raw,
        "evaluated_cases": len(selected_cases),
        "available_cases": len(manifest["cases"]),
        "sampling": (
            "independent Gaussian noise after downsampling; no OT during operational sampling"
        ),
        "limitations": (
            "Thresholded boundary/area/component diagnostics are not a complete topology "
            "or line-width evaluation."
        ),
        "results": results,
        "summary": summarize(results),
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(output / "metrics.json")


if __name__ == "__main__":
    main()
