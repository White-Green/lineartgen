"""Freeze a paired warmstart loss study and calibrate auxiliaries on training data only."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import sha, write
from flax import nnx

from lineartgen.config import save_config
from lineartgen.data.geometry import mix_geometry
from lineartgen.data.pipeline import batches, dataset_description, discover, split_experiment
from lineartgen.diffusion import recursive_forward, training_inputs
from lineartgen.models.serialization import load_inference
from lineartgen.training.lineart import line_region_loss, make_profiles, profile_loss
from lineartgen.training.lineart_adversarial import PatchCritic, generator_prior, patch_coordinates
from lineartgen.training.lineart_study import study_configs
from lineartgen.training.losses import image_loss


def prepare(root, initial_path, evaluation):
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(root)
    root.mkdir(parents=True)
    initial, original = load_inference(initial_path)
    if original.model.architecture != "unet":
        raise ValueError("Use the maximum baseline UNet")
    config = original.model_copy(
        update={
            "train": original.train.model_copy(
                update={
                    "steps": 10000,
                    "learning_rate": 1e-5,
                    "minimum_learning_rate": 1e-6,
                    "warmup_steps": 100,
                    "seed": 20261006,
                    "model_seed": 42,
                    "checkpoint_every": 250,
                    "log_every": 25,
                }
            )
        }
    )
    save_config(config, root / "config.json")
    sources = discover(config.data)
    train, dev, final = split_experiment(sources, config.data, config.train.seed)
    split = dataset_description(sources, train, dev, final)
    write(root / "dataset.json", split)
    manifest = json.loads((evaluation / "cases.json").read_text())
    assert manifest["dataset"] == split, "Evaluation split differs from training holdout"
    write(
        root / "evaluation.json",
        dict(
            root=str(evaluation.resolve()),
            cases_sha256=sha(evaluation / "cases.json"),
            files={c["file"]: sha(evaluation / c["file"]) for c in manifest["cases"]},
            role="Existing development data, not a new independent final test",
        ),
    )
    protocol = dict(
        initial=str(initial_path.resolve()),
        initial_files={p.name: sha(p) for p in initial_path.iterdir() if p.is_file()},
        milestones=[2000, 10000],
        patch_size=64,
        prior_ramp_steps=1000,
        primary_weights="ema",
        raw="sensitivity analysis",
        optimizer="Reset generator and critic optimizers; EMA starts at shared 100k EMA weights",
        fine_tuning="Base LR 1e-5 * sqrt(16) = 4e-5; cosine 10k, warmup 100, minimum 1e-6",
        critic_lr="Half generator base/minimum rates, same sqrt(batch) and schedule",
        calibration="Output-gradient norms at the frozen model, not parameter-gradient norms; "
        "training split only. 10% median / 50% maximum ratio, patch prior capped at .01.",
        selected_panels=sorted(
            set([4, 167, 256, 341, 479, 480, 1376, *range(0, len(manifest["cases"]), 64)])
        ),
        preservation="Same source/geometry/noise/mask series, recursion 1-6, scale-loss weight 0, "
        "same generator; geometry profiles only where exact paths are known.",
    )
    write(root / "protocol.json", protocol)
    # Calibrate at two batches per recursion depth, independently of their source scale.
    seed = 20262006
    stream = batches(
        train, config.data, batch_size=config.train.effective_batch_size, count=12, seed=seed
    )
    prepare_batch = jax.jit(
        lambda x, k, levels: training_inputs(
            x, k, config.diffusion, config.matching, levels=levels, sketch=config.sketch
        ),
        static_argnums=2,
    )
    infer = nnx.jit(recursive_forward)
    critic = PatchCritic(rngs=nnx.Rngs(config.train.initialization_seed + 5301))
    rows = []
    loss_config = config.loss.model_copy(update={"contrast_weight": 0.0})
    for step, batch in enumerate(stream):
        clean, records = mix_geometry(
            batch.images,
            config.geometry,
            seed=seed,
            step=step,
            image_scale=config.data.image_scale,
            include_paths=True,
        )
        profiles = jax.device_put(
            make_profiles(records, len(clean), clean.shape[1], seed=seed, step=step)
        )
        coords = jax.device_put(patch_coordinates(clean, seed=seed, step=step, size=64))
        levels = step // 2 + 1
        prepared = prepare_batch(
            jax.device_put(clean), jax.random.fold_in(jax.random.PRNGKey(seed), step), levels
        )
        prediction = infer(initial, prepared.noisy, prepared.noise_levels)[0]
        target, noise = prepared.targets[0], prepared.noise_levels[0]
        active = noise > config.diffusion.min_noise
        base_grad = jax.grad(
            lambda p, target=target, noise=noise: image_loss(
                p, target, noise, loss_config, config.diffusion.min_noise
            )[0]
        )(prediction)
        base_norm = float(np.linalg.norm(np.asarray(base_grad)))
        functions = dict(
            regions=lambda p, target=target, active=active: line_region_loss(
                p, target, active, config.loss.line_region_radius
            ),
            profiles=lambda p, target=target, active=active, profiles=profiles: profile_loss(
                p, target, active, profiles
            ),
            patch_prior=lambda p, prepared=prepared, coords=coords: generator_prior(
                critic, p, prepared, coords, 64, config.diffusion.min_noise
            ),
        )
        row = dict(
            batch=step,
            levels=levels,
            base_norm=base_norm,
            profile_count=float(np.asarray(profiles.valid).sum()),
            auxiliaries={},
        )
        for name, fn in functions.items():
            value, gradient = jax.value_and_grad(fn)(prediction)
            norm = float(np.linalg.norm(np.asarray(gradient)))
            if not np.isfinite(norm) or not np.isfinite(float(value)):
                raise FloatingPointError(name)
            row["auxiliaries"][name] = dict(
                value=float(value), norm=norm, ratio=norm / max(base_norm, 1e-12)
            )
        rows.append(row)
        print(json.dumps(row), flush=True)
    coefficients = {}
    for name in ("regions", "profiles", "patch_prior"):
        ratios = [
            r["auxiliaries"][name]["ratio"] for r in rows if r["auxiliaries"][name]["norm"] > 1e-12
        ]
        if not ratios:
            raise ValueError(f"No valid calibration gradient: {name}")
        coefficients[name] = float(
            min(
                0.1 / np.median(ratios),
                0.5 / max(ratios),
                0.01 if name == "patch_prior" else float("inf"),
            )
        )
    write(
        root / "calibration.json",
        dict(coefficients=coefficients, batches=rows, frozen_before_training=True),
    )
    (root / "configs").mkdir()
    for name, settings in study_configs(config, coefficients).items():
        save_config(settings, root / "configs" / f"{name}.json")
    print(json.dumps(dict(coefficients=coefficients)), flush=True)


def main():
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
    args = parser.parse_args()
    prepare(args.root.resolve(), args.initial, args.evaluation)


if __name__ == "__main__":
    main()
