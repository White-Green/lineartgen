"""Paired primary-loss intervention, with a diagnostic that never tunes coefficients."""

import argparse
import shutil
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from flax import nnx

from lineartgen.config import load_config, save_config
from lineartgen.data.geometry import mix_geometry
from lineartgen.data.pipeline import batches, dataset_description, discover, split_experiment
from lineartgen.diffusion import training_inputs
from lineartgen.models import create_model
from lineartgen.models.serialization import load_inference
from lineartgen.training.lineart import make_profiles
from lineartgen.training.lineart_adversarial import patch_coordinates
from lineartgen.training.lineart_steps import accumulate_generator, make_lineart_steps


def prepare(root, hard, continuation):
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(root)
    root.mkdir(parents=True)
    for name in (
        "dataset.json",
        "evaluation.json",
        "acceptance.json",
        "supplemental-panels.json",
        "confirmation-plan.json",
    ):
        shutil.copyfile(hard / name, root / name)
    configs, counterparts = {}, {}
    for counterpart in ("control", "ink_guard"):
        base = load_config(hard / f"configs/{counterpart}.json")
        paths = {
            name: (hard.parent.parent / getattr(base.data, name)).resolve()
            for name in ("root", "prepared_root")
            if getattr(base.data, name) is not None
        }
        base = base.model_copy(update={"data": base.data.model_copy(update=paths)})
        assert base.model.output_clip_gradient == "hard" and base.loss.pixel_loss == "clipped_mse"
        assert base.diffusion.scale_loss_multiplier == 0
        name = f"interval_{counterpart}"
        configs[name] = base.model_copy(
            update={"loss": base.loss.model_copy(update={"pixel_loss": "interval_mse"})}
        )
        counterparts[name] = dict(
            root=str(hard), method=counterpart, changes={"loss": {"pixel_loss": "interval_mse"}}
        )
    plan = dict(
        methods=list(configs),
        primary_error=(
            "clean==1: min(z-clean,0)^2; clean==-1: max(z-clean,0)^2; otherwise (z-clean)^2"
        ),
        physical_preclip=True,
        clipped_recursive_feedback=True,
        auxiliary_losses="Unchanged, evaluated on hard-clipped predictions with true derivatives",
        unchanged="100k EMA, data, seed, optimizer, stage weights, recursion, LR, gates",
        coefficients="No new coefficient and no normalization tuned from quality results",
        diagnostic=dict(
            seed=20262006,
            batches=12,
            depths=[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
            purpose="Measure parameter-gradient change, not tune or update weights",
        ),
        selection="Inherited 2k screen and visual review; up to two passing arms to full-state 10k",
        confirmation="Final-gate passing candidates receive second-seed and sealed fresh tests",
        changed_training_files=[
            "src/lineartgen/diffusion.py",
            "src/lineartgen/training/steps.py",
            "src/lineartgen/training/lineart_steps.py",
        ],
    )
    write(root / "plan.json", plan)
    original = read(hard / "protocol.json")
    external = {
        str(step): [
            dict(
                root=str(parent),
                methods=["control", "ink_guard"],
                source_sha256=sha(parent / "source.json"),
            )
        ]
        for step, parent in [(2000, hard), (10000, continuation)]
    }
    write(
        root / "protocol.json",
        original
        | dict(
            methods=list(configs),
            milestones=[2000],
            external_baselines=external,
            config_counterparts=counterparts,
            plan_sha256=sha(root / "plan.json"),
            calibration="No loss coefficient calibration; 12 training-only gradient comparisons",
            optimizer="Reset optimizer/EMA from common 100k EMA; unchanged 10k cosine clock",
            next_phase=(
                "Apply inherited 2k screen and visual review before full-state 10k continuation"
            ),
        ),
    )
    base = configs["interval_control"]
    save_config(base, root / "config.json")
    (root / "configs").mkdir()
    for name, config in configs.items():
        save_config(config, root / f"configs/{name}.json")
    training, dev, final = split_experiment(discover(base.data), base.data, base.train.seed)
    assert dataset_description(discover(base.data), training, dev, final) == read(
        root / "dataset.json"
    )
    print(dict(root=str(root), methods=list(configs)), flush=True)


def diagnose(root):
    if (root / "calibration.json").exists():
        raise FileExistsError(root / "calibration.json")
    plan, protocol = read(root / "plan.json"), read(root / "protocol.json")
    assert sha(root / "plan.json") == protocol["plan_sha256"]
    diagnostic = plan["diagnostic"]
    base = load_config(root / "config.json")
    initial, _ = load_inference(protocol["initial"])
    model = create_model(base.model, rngs=nnx.Rngs(base.train.initialization_seed))
    nnx.update(model, nnx.state(initial, nnx.Param))
    before = jax.tree.map(lambda x: np.asarray(x).copy(), nnx.state(model))
    configs, gradients = {}, {}
    for name in protocol["methods"]:
        config = load_config(root / f"configs/{name}.json")
        configs[name] = config
        configs[name.removeprefix("interval_")] = config.model_copy(
            update={"loss": config.loss.model_copy(update={"pixel_loss": "clipped_mse"})}
        )
    for name, config in configs.items():
        gradients[name] = make_lineart_steps(config)[0]
    training, _, _ = split_experiment(discover(base.data), base.data, base.train.seed)
    seed = diagnostic["seed"]
    stream = batches(
        training,
        base.data,
        batch_size=base.train.effective_batch_size,
        count=diagnostic["batches"],
        seed=seed,
    )
    prepare_inputs = jax.jit(
        lambda x, k, n: training_inputs(
            x, k, base.diffusion, base.matching, levels=n, sketch=base.sketch
        ),
        static_argnums=2,
    )
    rows = []
    try:
        for step, batch in enumerate(stream):
            clean, geometry = mix_geometry(
                batch.images,
                base.geometry,
                seed=seed,
                step=step,
                image_scale=base.data.image_scale,
                include_paths=True,
            )
            profiles, coordinates = jax.device_put(
                (
                    make_profiles(geometry, len(clean), len(clean[0]), seed=seed, step=step),
                    patch_coordinates(clean, seed=seed, step=step, size=protocol["patch_size"]),
                )
            )
            levels = diagnostic["depths"][step]
            prepared = prepare_inputs(
                jax.device_put(clean), jax.random.fold_in(jax.random.PRNGKey(seed), step), levels
            )
            measures = {}
            for name in configs:
                metrics, _, norm = accumulate_generator(
                    gradients[name], model, prepared, profiles, coordinates, None, 0, base.train
                )
                measures[name] = dict(loss=float(metrics["loss"]), norm=float(norm))
                assert all(np.isfinite(v) for v in measures[name].values())
            row = dict(
                batch=step,
                levels=levels,
                measures=measures,
                norm_ratios={
                    name: measures[name]["norm"]
                    / max(measures[name.removeprefix("interval_")]["norm"], 1e-12)
                    for name in protocol["methods"]
                },
            )
            rows.append(row)
            print(row, flush=True)
    finally:
        stream.close()
    jax.tree.map(np.testing.assert_array_equal, before, nnx.state(model))
    write(
        root / "calibration.json",
        dict(
            coefficients={},
            diagnostic_only=True,
            no_model_updates=True,
            rows=rows,
            plan_sha256=sha(root / "plan.json"),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "diagnose"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--hard", type=Path)
    parser.add_argument("--continuation", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.hard is None or args.continuation is None:
            parser.error("prepare requires --hard and --continuation")
        prepare(args.root.resolve(), args.hard.resolve(), args.continuation.resolve())
    else:
        diagnose(args.root.resolve())
