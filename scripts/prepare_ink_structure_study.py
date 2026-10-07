"""Prepare a six-arm structural-loss screen; calibrate using training images only."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from flax import nnx

from lineartgen.config import save_config
from lineartgen.data.geometry import mix_geometry
from lineartgen.data.pipeline import batches, dataset_description, discover, split_experiment
from lineartgen.diffusion import recursive_forward, training_inputs
from lineartgen.models.serialization import load_inference
from lineartgen.training.ink_structure import ink_structure_losses
from lineartgen.training.losses import image_loss

METHODS = ("control", "no_contrast", "regions", "weak_regions", "ink_guard", "neighbor_guard")


def prepare(root, initial_path, evaluation, previous, seed):
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(root)
    root.mkdir(parents=True)
    initial, original = load_inference(initial_path)
    assert original.model.architecture == "unet"
    config = original.model_copy(
        update={
            "train": original.train.model_copy(
                update={
                    "steps": 10000,
                    "learning_rate": 1e-5,
                    "minimum_learning_rate": 1e-6,
                    "warmup_steps": 100,
                    "seed": seed,
                    "model_seed": 42,
                    "checkpoint_every": 250,
                    "log_every": 25,
                }
            )
        }
    )
    save_config(config, root / "config.json")
    sources = discover(config.data)
    train, dev, final = split_experiment(sources, config.data, seed)
    split = dataset_description(sources, train, dev, final)
    write(root / "dataset.json", split)
    manifest = read(evaluation / "cases.json")
    assert manifest["dataset"] == split
    write(
        root / "evaluation.json",
        dict(
            root=str(evaluation.resolve()),
            cases_sha256=sha(evaluation / "cases.json"),
            files={c["file"]: sha(evaluation / c["file"]) for c in manifest["cases"]},
            role="Reused development for screening; fresh synthetic confirmation is separate",
        ),
    )
    supplemental = read(previous / "supplemental-panels.json")
    assert supplemental["manifest_sha256"] == sha(evaluation / "cases.json")
    write(root / "supplemental-panels.json", supplemental)
    old_protocol = read(previous / "protocol.json")
    region_weight = read(previous / "calibration.json")["coefficients"]["regions"]
    assert old_protocol["initial_files"]["weights.npz"] == sha(initial_path / "weights.npz")
    panels = sorted(
        set(old_protocol["selected_panels"] + [r["case"] for r in supplemental["cases"]])
    )
    write(
        root / "protocol.json",
        dict(
            methods=list(METHODS),
            initial=str(initial_path.resolve()),
            initial_files={p.name: sha(p) for p in initial_path.iterdir() if p.is_file()},
            milestones=[2000],
            total_schedule_steps=10000,
            patch_size=64,
            prior_ramp_steps=1000,
            primary_weights="ema",
            selected_panels=panels,
            optimizer="All arms reset optimizer, common initial 100k EMA; cosine clock 10k",
            calibration="12 training batches; output-gradient norm ratios, not parameter gradients",
            region_coefficient_source=str(previous / "calibration.json"),
            region_coefficient_sha256=sha(previous / "calibration.json"),
            region_weight=region_weight,
            weak_region_multiplier=0.25,
            next_phase="Inspect 2k; fork full optimizer/EMA states for control "
            "and at most two candidates",
            preservation="Common data/noise/masks, recursion 1-6, stage-loss weights, "
            "sqrt(batch) peak LR, Krita IO",
        ),
    )
    write(
        root / "acceptance.json",
        dict(
            primary_baseline="control",
            attribution_baseline="no_contrast",
            halo_groups=["real/strokes/strong", "geometry/strokes/strong"],
            minimum_relative_halo_gain=0.10,
            safety_groups=[
                f"{d}/{f}/{s}"
                for d in ["real", "geometry"]
                for f, s in [
                    ("clean", "clean"),
                    ("strokes", "mild"),
                    ("strokes", "strong"),
                    ("blur", "mild"),
                    ("blur", "strong"),
                    ("tone", "mild"),
                    ("tone", "strong"),
                ]
            ],
            maximum_increases={
                "ink_deficit_fraction": 0.005,
                "profile_missing_fraction": 0.005,
                "connection_cut_rate": 0.01,
                "connection_bridge_rate": 0.01,
                "profile_width_mae_px": 0.10,
                "profile_offset_mae_px": 0.05,
                "far_opacity_excess": 0.0002,
            },
            maximum_active_f1_drop=0.005,
            required_strata=["primary", "strength-0.1", "partial", "whole"],
            screen_is_not_acceptance=True,
            screening="2k numerical/visual review. Keep at most two non-control candidates with "
            "at least 5% halo gain in one primary group, <=2pp ink-deficit increase, "
            "<=3pp cut/bridge/missing increase in safety groups; rank by mean primary halo gain.",
            confirmation="10k primary EMA and raw sensitivity; visual review; same criteria on new "
            "synthetic corruption seeds and a second independent training seed. Report "
            "source-cluster CIs, including whether primary halo intervals exclude zero. "
            "Do not pool diagnostics to conceal per-domain regressions.",
            limitation="Practical point-estimate gates, not a mathematical guarantee "
            "of topology or subjective satisfaction",
        ),
    )
    calibration_seed = 20262006
    prepare_batch = jax.jit(
        lambda x, k, levels: training_inputs(
            x, k, config.diffusion, config.matching, levels=levels, sketch=config.sketch
        ),
        static_argnums=2,
    )
    infer = nnx.jit(recursive_forward)
    base_loss = config.loss.model_copy(update={"contrast_weight": 0.0})
    rows = []
    stream = batches(
        train,
        config.data,
        batch_size=config.train.effective_batch_size,
        count=12,
        seed=calibration_seed,
    )
    try:
        for step, batch in enumerate(stream):
            clean, _ = mix_geometry(
                batch.images,
                config.geometry,
                seed=calibration_seed,
                step=step,
                image_scale=config.data.image_scale,
            )
            levels = step // 2 + 1
            prepared = prepare_batch(
                jax.device_put(clean),
                jax.random.fold_in(jax.random.PRNGKey(calibration_seed), step),
                levels,
            )
            prediction = infer(initial, prepared.noisy, prepared.noise_levels)[0]
            target, noise = prepared.targets[0], prepared.noise_levels[0]
            active = noise > config.diffusion.min_noise
            base_gradient = jax.grad(
                lambda p, target=target, noise=noise: image_loss(
                    p, target, noise, base_loss, config.diffusion.min_noise
                )[0]
            )(prediction)
            bg = np.asarray(base_gradient)
            norm = float(np.linalg.norm(bg))
            row = dict(batch=step, levels=levels, base_norm=norm, auxiliaries={})
            for name in ("ink_keep", "neighbor_keep", "neighbor_paper"):
                value, gradient = jax.value_and_grad(
                    lambda p, target=target, active=active, name=name: ink_structure_losses(
                        p, target, active
                    )[name]
                )(prediction)
                g = np.asarray(gradient)
                gnorm = float(np.linalg.norm(g))
                assert np.isfinite(gnorm) and np.isfinite(float(value))
                row["auxiliaries"][name] = dict(
                    value=float(value),
                    norm=gnorm,
                    ratio=gnorm / max(norm, 1e-12),
                    output_gradient_cosine=float(np.sum(bg * g) / max(norm * gnorm, 1e-12)),
                )
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        stream.close()
    coefficients = {}
    for name in ("ink_keep", "neighbor_keep", "neighbor_paper"):
        ratios = [
            r["auxiliaries"][name]["ratio"] for r in rows if r["auxiliaries"][name]["norm"] > 1e-12
        ]
        if not ratios:
            raise ValueError(f"No calibration gradient for {name}")
        desired, cap = (0.05, 0.25) if name == "neighbor_paper" else (0.2, 0.75)
        coefficients[name] = float(min(desired / np.median(ratios), cap / max(ratios)))
    write(
        root / "calibration.json",
        dict(coefficients=coefficients, batches=rows, frozen_before_training=True),
    )
    losses = dict(
        control=config.loss,
        no_contrast=base_loss,
        regions=base_loss.model_copy(update={"line_region_weight": region_weight}),
        weak_regions=base_loss.model_copy(update={"line_region_weight": region_weight / 4}),
        ink_guard=base_loss.model_copy(
            update={
                "line_region_weight": region_weight,
                "ink_keep_weight": coefficients["ink_keep"],
            }
        ),
        neighbor_guard=base_loss.model_copy(
            update={
                "line_region_weight": region_weight,
                "neighbor_keep_weight": coefficients["neighbor_keep"],
                "neighbor_paper_weight": coefficients["neighbor_paper"],
            }
        ),
    )
    (root / "configs").mkdir()
    for name, loss in losses.items():
        save_config(config.model_copy(update={"loss": loss}), root / "configs" / f"{name}.json")
    print(json.dumps(dict(coefficients=coefficients)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--initial",
        type=Path,
        default=Path(
            "runs/completion-20260919-cached/arms/seed42_baseline/snapshots/step-100000/model/ema"
        ),
    )
    parser.add_argument(
        "--evaluation", type=Path, default=Path("runs/geometry-data-20260914/evaluation")
    )
    parser.add_argument("--previous", type=Path, default=Path("runs/lineart-loss-20261006"))
    parser.add_argument("--seed", type=int, default=20261007)
    args = parser.parse_args()
    prepare(args.root.resolve(), args.initial, args.evaluation, args.previous, args.seed)
