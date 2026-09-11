"""Evaluate all three widths of each method on fixed matched and geometric cases."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import jax
import numpy as np
from evaluate_nested import geometry_metrics
from flax import nnx
from PIL import Image, ImageDraw

from lineartgen.config import load_config
from lineartgen.data.images import read_crop
from lineartgen.data.pipeline import dataset_description, discover, split_experiment
from lineartgen.diffusion import sample
from lineartgen.evaluation import KINDS, line_case, probe_metrics, protected_metrics
from lineartgen.models.serialization import load_inference, save_inference
from lineartgen.training.checkpoints import load_run_model
from lineartgen.training.comparison import comparison_configs


def data_settings(config):
    return config.data.model_dump(
        mode="json", exclude={"workers", "prefetch_batches", "prepared_root"}
    )


def prepare(config, output, legacy):
    manifest_path = output / "cases.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing["data_config"] != data_settings(config):
            raise ValueError("Evaluation data settings changed")
        return existing
    output.mkdir(parents=True, exist_ok=True)
    sources = discover(config.data)
    training, development, final = split_experiment(sources, config.data, config.train.seed)
    description = dataset_description(sources, training, development, final)
    old = json.loads((legacy / "cases.json").read_text())
    if old["dataset"] != description:
        raise ValueError("Legacy cases use a different split")
    records = []
    for i, case in enumerate(old["cases"]):
        name = f"case-{i:03d}.npz"
        shutil.copyfile(legacy / case["file"], output / name)
        records.append({**case, "file": name, "suite": "legacy"})

    def store(image, strength, partial, record, probes=None, connections=None):
        index = len(records)
        noise = np.full_like(image, strength)
        if partial:
            noise[:102] = 0
            noise[:, :170] = 0
            noise[384:, 384:] = 0
        name = f"case-{index:03d}.npz"
        arrays = dict(image=image, strength=noise)
        if probes is not None:
            arrays.update(probes=probes, connections=connections)
        np.savez_compressed(output / name, **arrays)
        records.append(
            dict(record, file=name, strength=strength, partial=partial, seed=2026091100 + index)
        )

    def normalize(gray):
        return np.clip((gray * 2 - 1) * config.data.image_scale, -1, 1).astype(np.float32)

    # Two line-containing crops per scale, each reused for all nine conditions.
    # Selection depends only on targets; retain the previous blank-crop cases too.
    random = np.random.default_rng(20260911)
    for variant, level in enumerate(config.data.resolved_scale_levels):
        chosen = set()
        for crop_index in range(2):
            for _ in range(128):
                source = development[int(random.integers(len(development)))]
                if source.name in chosen:
                    continue
                path, width, height = source.variants[variant]
                x, y = int(random.integers(width - 512 + 1)), int(random.integers(height - 512 + 1))
                image = normalize(read_crop(path, x, y, 512, alpha_mode=config.data.alpha_mode))
                if (image < 0).mean() >= 0.001:
                    break
            else:
                raise RuntimeError(f"No line-containing crop found at scale {level}")
            chosen.add(source.name)
            for i in range(9):
                strength = [0.1, 0.3, 0.5, 0.7][i % 4] if i < 8 else 0.0
                store(
                    image,
                    strength,
                    4 <= i < 8,
                    dict(
                        kind="real",
                        suite="matched_real",
                        source=source.name,
                        crop=[x, y, 512],
                        scale_level=level,
                        matched_crop=f"{level}-{crop_index}",
                    ),
                )
    for kind in KINDS:
        for variant in range(8):
            gray, probes, connections = line_case(kind, variant)
            image = normalize(gray)
            for i, strength in enumerate([0.1, 0.3, 0.5, 0.7]):
                if i == 0 and variant % 4 == 0:
                    strength = 0.0
                store(
                    image,
                    strength,
                    i >= 2,
                    dict(kind=kind, suite="geometry", variant=variant, scale_level=0),
                    probes,
                    connections,
                )
    manifest = dict(
        format_version=1, dataset=description, data_config=data_settings(config), cases=records
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(records)} cases: legacy 70 + matched real 108 + geometry 256", flush=True)
    return manifest


def summarize(rows):
    subsets = {
        "legacy_synthetic": lambda r: r["suite"] == "legacy" and r["kind"] != "real",
        "legacy_real_with_lines": lambda r: (
            r["suite"] == "legacy"
            and r["kind"] == "real"
            and r["strength"] > 0
            and r["target_ink_fraction"] >= 0.001
        ),
        "matched_real": lambda r: r["suite"] == "matched_real" and r["strength"] > 0,
        "matched_real_active_lines": lambda r: (
            r["suite"] == "matched_real"
            and r["strength"] > 0
            and r["active_target_ink_fraction"] >= 0.001
        ),
        "geometry": lambda r: r["suite"] == "geometry" and r["strength"] > 0,
        "real_identity": lambda r: r["kind"] == "real" and r["strength"] == 0,
        "geometry_identity": lambda r: r["suite"] == "geometry" and r["strength"] == 0,
        "blank_real": lambda r: (
            r["kind"] == "real" and r["strength"] > 0 and r["target_ink_fraction"] < 0.001
        ),
    }
    metrics = [
        "active_mae",
        "active_ink_f1",
        "mae",
        "ink_f1",
        "boundary_f1_1_5px",
        "protected_mae",
        "protected_max_error",
        "mask_boundary_mae",
        "profile_width_mae_px",
        "profile_missing_fraction",
        "profile_offset_mae_px",
        "profile_offset_variation_px",
        "connection_cut_rate",
        "connection_bridge_rate",
    ]
    summaries = []
    for name, select in subsets.items():
        for method, subnet, steps in sorted(
            {(r["method"], r["subnet"], r["denoise_steps"]) for r in rows}
        ):
            selected = [
                r
                for r in rows
                if select(r)
                and (r["method"], r["subnet"], r["denoise_steps"]) == (method, subnet, steps)
            ]
            if not selected:
                continue
            result = dict(
                subset=name, method=method, subnet=subnet, denoise_steps=steps, cases=len(selected)
            )
            for metric in metrics:
                values = [r[metric] for r in selected if r.get(metric) is not None]
                result[metric] = float(np.mean(values)) if values else None
            # Max of per-case maxima matters for a structural preservation claim.
            values = [
                r["protected_max_error"] for r in selected if r["protected_max_error"] is not None
            ]
            result["protected_worst_error"] = float(max(values)) if values else None
            summaries.append(result)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/nested.toml"))
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument(
        "--legacy-cases", type=Path, default=Path("runs/nested-20260910/evaluation")
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run-group", type=Path)
    parser.add_argument("--step", type=int, required=False)
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 5])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--snapshot-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.data.crop_size != 512:
        parser.error("This fixed evaluation suite uses 512px crops")
    manifest = prepare(config, args.cases, args.legacy_cases)
    if args.prepare_only:
        return
    if args.run_group is None or args.step is None:
        parser.error("--run-group and --step are required for evaluation")
    if json.loads((args.run_group / "dataset.json").read_text()) != manifest["dataset"]:
        raise ValueError("Evaluation and training use different data or splits")
    if any(x < 1 for x in args.steps) or (args.limit is not None and args.limit < 1):
        parser.error("steps and limit must be positive")
    mode = "raw" if args.raw else "ema"
    suffix = f"-preflight-{args.limit}" if args.limit is not None else ""
    output = args.cases / f"step-{args.step}-{mode}{suffix}"
    output.mkdir(exist_ok=True)
    models = {}
    for method, settings in comparison_configs(config, study="methods").items():
        snapshot = args.run_group.parent / "snapshots" / f"step-{args.step}" / method / mode
        if snapshot.exists():
            model, saved = load_inference(snapshot)
        else:
            model, saved, step = load_run_model(
                args.run_group / method, step=args.step, use_ema=not args.raw
            )
            assert step == args.step
            save_inference(model, saved, snapshot, provenance=f"{method} step {step} {mode}")
        if saved != settings:
            raise ValueError(f"Saved configuration differs for {method}")
        models.update(((method, i), model.extract(i)) for i in range(model.num_subnets))
    if args.snapshot_only:
        print(f"Saved {mode} snapshots at step {args.step}", flush=True)
        return
    run_sample = nnx.jit(
        lambda model, image, noise, key, steps, levels: sample(
            model, image, key, strength=noise, steps=steps, levels=levels
        ),
        static_argnums=(4, 5),
    )
    results = []
    selected_cases = manifest["cases"][: args.limit]
    for index, case in enumerate(selected_cases):
        with np.load(args.cases / case["file"], allow_pickle=False) as arrays:
            target, strength = arrays["image"], arrays["strength"]
            probes = arrays["probes"] if "probes" in arrays else None
            connections = arrays["connections"] if "connections" in arrays else None
        image, noise = jax.device_put((target[None], strength[None]))
        levels = config.diffusion.levels_for_scale(
            512, config.model.minimum_size, case["scale_level"]
        )
        key = jax.random.PRNGKey(case["seed"])
        display_key = key if levels == 1 else jax.random.split(key, levels)[0]
        noisy = np.asarray(
            image * jax.numpy.sqrt(1 - noise)
            + jax.random.normal(display_key, image.shape) * jax.numpy.sqrt(noise)
        )[0]
        for steps in args.steps:
            pictures = [target, noisy]
            labels = ["target", "noisy input"]
            for (method, subnet), model in models.items():
                prediction = np.asarray(run_sample(model, image, noise, key, steps, levels))[0]
                extra = (
                    probe_metrics(prediction, target, probes, connections)
                    if probes is not None
                    else {}
                )
                results.append(
                    dict(
                        case=index,
                        kind=case["kind"],
                        suite=case["suite"],
                        source=case.get("source"),
                        scale_level=case["scale_level"],
                        strength=case["strength"],
                        partial=case["partial"],
                        method=method,
                        subnet=subnet,
                        denoise_steps=steps,
                        **geometry_metrics(prediction, target, strength),
                        **protected_metrics(prediction, target, strength),
                        **extra,
                    )
                )
                if subnet == 0:
                    pictures.append(prediction)
                    labels.append(method + " S")
            if index < 70 or index % 16 == 0:
                panel = np.concatenate(pictures, axis=1)[..., 0]
                pixels = np.rint(np.clip((panel + 1) * 127.5, 0, 255)).astype(np.uint8)
                canvas = Image.new("L", (pixels.shape[1], pixels.shape[0] + 24), 255)
                canvas.paste(Image.fromarray(pixels), (0, 24))
                draw = ImageDraw.Draw(canvas)
                for column, label in enumerate(labels):
                    draw.text((column * 512 + 8, 5), label, fill=0)
                canvas.save(output / f"case-{index:03d}-steps-{steps}.png")
        if (index + 1) % 10 == 0 or index + 1 == len(selected_cases):
            print(f"Evaluated {mode} case {index + 1}/{len(selected_cases)}", flush=True)
    report = dict(
        checkpoint_step=args.step,
        ema=not args.raw,
        cases=len(selected_cases),
        available_cases=len(manifest["cases"]),
        manifest_sha256=hashlib.sha256((args.cases / "cases.json").read_bytes()).hexdigest(),
        sampling="Independent Gaussian after downsampling; no OT",
        results=results,
        summary=summarize(results),
        limitations=(
            "Normal-profile and thresholded connection probes are diagnostics, not complete "
            "topology/curvature measurements. Connection queries invalid in the rasterized "
            "target are excluded; visibility is reported separately from position error."
        ),
    )
    temporary = output / "metrics.partial.json"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "metrics.json")
    print(output / "metrics.json", flush=True)


if __name__ == "__main__":
    main()
