"""Paired training of nested widths or four learning-method ablations.

Optionally add the original JAX UNet as a quality reference. All runs consume
the same source crops. Methods use a common Gaussian pool before assignment. No model shares
optimizer state with another run. Checkpoints use the ordinary Trainer format.
"""

import argparse
import hashlib
import json
import math
import time
from contextlib import ExitStack
from pathlib import Path

import jax
import numpy as np
from flax import nnx
from tensorboardX import SummaryWriter

from lineartgen.config import load_config, save_config
from lineartgen.data.pipeline import batches, dataset_description, discover, split_experiment
from lineartgen.diffusion import sample, training_inputs
from lineartgen.models import create_model
from lineartgen.models.serialization import load_inference, save_inference
from lineartgen.training import checkpoints
from lineartgen.training.comparison import comparison_configs
from lineartgen.training.steps import accumulate_gradients, create_optimizer, make_steps
from lineartgen.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/nested.toml"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stop-after", type=int, default=2000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument(
        "--study", choices=["widths", "methods", "sketch", "sketch_rates"], default="widths"
    )
    parser.add_argument(
        "--initialize-from", type=Path, help="Shared inference weights; reset optimizer and EMA"
    )
    parser.add_argument(
        "--stop-file",
        type=Path,
        help="Save all runs and stop at the next complete update when this file exists",
    )
    args = parser.parse_args()
    if args.stop_after < 1:
        parser.error("--stop-after must be positive")
    if args.initialize_from and (args.resume or args.study == "widths"):
        parser.error("--initialize-from requires a fresh methods/sketch study")
    config = load_config(args.config)
    configs = comparison_configs(config, study=args.study, baseline=args.baseline)
    for candidate in configs.values():
        for scale in config.data.resolved_scale_levels:
            if candidate.diffusion.levels_for_scale(
                candidate.data.crop_size, candidate.model.minimum_size, scale
            ) != config.diffusion.levels_for_scale(
                config.data.crop_size, config.model.minimum_size, scale
            ):
                raise ValueError("Paired runs require the same recursion count at every scale")
    root = args.run_dir.resolve()
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise FileExistsError(f"Run group is not empty: {root}; use --resume")
    root.mkdir(parents=True, exist_ok=True)
    initial = None
    if args.initialize_from:
        initial, initial_config = load_inference(args.initialize_from)
        if any(c.model != initial_config.model for c in configs.values()):
            raise ValueError("Warmstart architecture differs from the comparison")
        (root / "initialization.json").write_text(
            json.dumps(
                {
                    "source": str(args.initialize_from.resolve()),
                    "files": {
                        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in args.initialize_from.iterdir()
                        if p.is_file()
                    },
                    "optimizer": "reset",
                    "ema": "copy initial parameters",
                },
                indent=2,
            )
            + "\n"
        )
    progress = json.loads((root / "progress.json").read_text()) if args.resume else None
    if progress is not None and progress["runs"] != list(configs):
        raise ValueError("Resume must include the same runs (including --baseline)")
    sources = discover(config.data)
    training, validation, final_test = split_experiment(sources, config.data, config.train.seed)
    description = dataset_description(sources, training, validation, final_test)
    if args.resume and json.loads((root / "dataset.json").read_text()) != description:
        raise ValueError("Dataset or split changed since the run group was created")
    (root / "dataset.json").write_text(json.dumps(description, indent=2) + "\n")
    print(
        f"Sources: train={len(training)}, dev={len(validation)}, final={len(final_test)}",
        flush=True,
    )
    completed = progress["step"] if progress is not None else 0
    end = min(config.train.steps, completed + args.stop_after)
    with ExitStack() as stack:
        runs = {}
        # These comparisons preserve the loss and optimizer settings. Reuse the
        # numerical functions so identical architectures share compiled programs.
        gradient, update, evaluate = make_steps(config)
        preparations = {}

        def preparation(settings):
            key = (settings.diffusion, settings.matching, settings.sketch)
            if key not in preparations:
                preparations[key] = jax.jit(
                    lambda x, rng, levels: training_inputs(
                        x,
                        rng,
                        settings.diffusion,
                        settings.matching,
                        levels=levels,
                        sketch=settings.sketch,
                    ),
                    static_argnums=(2,),
                )
            return preparations[key]

        for name, settings in configs.items():
            directory = root / name
            directory.mkdir(exist_ok=True)
            if args.resume:
                if load_config(directory / "config.json") != settings:
                    raise ValueError(f"Configuration changed for {name}")
                if json.loads((directory / "dataset.json").read_text()) != description:
                    raise ValueError(f"Dataset changed for {name}")
            else:
                save_config(settings, directory / "config.json")
                (directory / "dataset.json").write_text(json.dumps(description, indent=2) + "\n")
            model = create_model(settings.model, rngs=nnx.Rngs(settings.train.initialization_seed))
            if initial is not None:
                nnx.update(model, jax.tree.map(lambda x: x.copy(), nnx.state(initial, nnx.Param)))
                save_inference(
                    model,
                    settings,
                    root / "initial" / name,
                    provenance=f"Shared warmstart from {args.initialize_from}",
                )
            optimizer, schedule = create_optimizer(model, settings)
            ema = jax.tree.map(lambda x: x.copy(), nnx.state(model, nnx.Param))
            manager = stack.enter_context(
                checkpoints.manager(directory / "checkpoints", settings.train.keep_checkpoints)
            )
            if args.resume:
                restored, ema = checkpoints.restore(manager, model, optimizer, ema, completed)
                assert restored == completed
            writer = stack.enter_context(
                SummaryWriter(
                    str(directory / "events"), purge_step=completed if args.resume else None
                )
            )
            runs[name] = dict(
                config=settings,
                model=model,
                optimizer=optimizer,
                ema=ema,
                saved=manager,
                writer=writer,
                gradient=gradient,
                update=update,
                evaluate=evaluate,
                schedule=schedule,
                trainer=Trainer(settings, directory),
                prepare=preparation(settings),
            )
        if not args.resume:
            # The contrast starts with exactly the same weights, not just the same seed.
            if args.study == "widths":
                pairs = [(runs["shared"]["model"].extract(0), runs["independent_core"]["model"])]
            else:
                first = next(iter(runs.values()))["model"]
                pairs = [(first, run["model"]) for run in runs.values()]
            for a, b in pairs:
                jax.tree.map(
                    np.testing.assert_array_equal, nnx.state(a, nnx.Param), nnx.state(b, nnx.Param)
                )
        validation_prepare = preparation(
            configs["independent_noise"]
            if args.study == "methods"
            else configs["strokes"]
            if args.study == "sketch"
            else configs["p75"]
            if args.study == "sketch_rates"
            else config
        )
        sampler = nnx.jit(
            lambda model, clean, key, levels: sample(
                model, clean, key, strength=0.5, steps=config.diffusion.sample_steps, levels=levels
            ),
            static_argnums=(3,),
        )
        stream = batches(
            training,
            config.data,
            batch_size=config.train.effective_batch_size,
            count=end - completed,
            seed=config.train.seed,
            start=completed,
        )
        stack.callback(stream.close)
        print(f"Devices: {jax.devices()}; paired steps {completed} → {end}", flush=True)
        root_key = jax.random.PRNGKey(config.train.seed)
        for step in range(completed, end):
            started = time.perf_counter()
            batch = next(stream)
            clean = jax.device_put(batch.images)
            jax.block_until_ready(clean)
            input_seconds = time.perf_counter() - started
            levels = config.diffusion.levels_for_scale(
                config.data.crop_size, config.model.minimum_size, batch.scale_level
            )
            prepared_at = time.perf_counter()
            prepared_batches = {
                key: prepare(clean, jax.random.fold_in(root_key, step), levels)
                for key, prepare in preparations.items()
            }
            jax.block_until_ready(tuple(prepared_batches.values()))
            preparation_seconds = time.perf_counter() - prepared_at
            timings = {}
            # Rotate through every position, including comparisons with four runs.
            names = list(runs)
            offset = step % len(names)
            order = names[offset:] + names[:offset]
            for name in order:
                run = runs[name]
                settings = run["config"]
                prepared = prepared_batches[
                    (settings.diffusion, settings.matching, settings.sketch)
                ]
                update_at = time.perf_counter()
                metrics, gradients, norm = accumulate_gradients(
                    run["gradient"], run["model"], prepared, config.train
                )
                norm = float(norm)
                if not math.isfinite(norm) or not math.isfinite(float(metrics["loss"])):
                    raise FloatingPointError(f"Nonfinite loss/gradient: {name} at {step + 1}")
                _, run["ema"] = run["update"](run["model"], run["optimizer"], gradients, run["ema"])
                jax.block_until_ready((nnx.state(run["model"]), run["ema"]))
                timings[name] = time.perf_counter() - update_at
                if (step + 1) % config.train.log_every == 0 or step + 1 == end:
                    logged = {key: float(value) for key, value in metrics.items()}
                    logged.update(
                        gradient_norm=norm,
                        learning_rate=float(run["schedule"](step)),
                        scale_level=batch.scale_level,
                        recursion_levels=levels,
                        update_seconds=timings[name],
                    )
                    run["trainer"]._log(run["writer"], step + 1, "train", logged)
            with (root / "timings.jsonl").open("a") as output:
                output.write(
                    json.dumps(
                        {
                            "step": step + 1,
                            "scale_level": batch.scale_level,
                            "recursion_levels": levels,
                            "input_seconds": input_seconds,
                            "preparation_seconds": preparation_seconds,
                            "updates": timings,
                            "seconds_before_evaluation": time.perf_counter() - started,
                        }
                    )
                    + "\n"
                )
            stop_requested = args.stop_file is not None and args.stop_file.exists()
            if (step + 1) % config.train.evaluate_every == 0 or step + 1 == end or stop_requested:
                for name, run in runs.items():
                    print(f"Evaluate {name} at {step + 1}", flush=True)
                    model = nnx.clone(run["model"])
                    nnx.update(model, run["ema"])
                    run["trainer"]._evaluate(
                        run["writer"],
                        step + 1,
                        model,
                        validation,
                        run["evaluate"],
                        sampler,
                        validation_prepare,
                    )
            if (step + 1) % config.train.checkpoint_every == 0 or step + 1 == end or stop_requested:
                for run in runs.values():
                    checkpoints.save(
                        run["saved"], run["model"], run["optimizer"], run["ema"], step + 1
                    )
                # Commit the group position only after every model checkpoint exists.
                temporary = root / "progress.partial.json"
                temporary.write_text(json.dumps({"step": step + 1, "runs": list(runs)}) + "\n")
                temporary.replace(root / "progress.json")
                print(f"Saved all runs at {step + 1}", flush=True)
            if stop_requested:
                end = step + 1
                break
    print(f"Completed paired step {end}", flush=True)


if __name__ == "__main__":
    main()
