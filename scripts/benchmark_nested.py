"""Measure shared nested updates and independent prefix updates on the same batch.

Example: JAX_PLATFORMS=rocm,cpu python scripts/benchmark_nested.py \
  --mode fused --levels 6 --batch-file runs/training-speed-20260909/batch16-512.f32
Input I/O is excluded; noise preparation, all microbatches and optimizer updates
are included. Prefixes are independently optimized only in `separate` mode.
"""

import argparse
import json
import statistics
import time

import jax
import numpy as np
from flax import nnx

from lineartgen.config import load_config
from lineartgen.diffusion import recursive_forward_all, training_inputs
from lineartgen.models import create_model
from lineartgen.training.steps import accumulate_gradients, create_optimizer, make_steps


def parameter_count(model):
    return sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/nested.toml")
    parser.add_argument(
        "--mode", choices=["fused", "blocks", "separate", "core", "baseline"], required=True
    )
    parser.add_argument("--levels", type=int, choices=range(1, 7), required=True)
    parser.add_argument("--batch-file", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--measure", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.mode == "baseline":
        config = config.model_copy(update={"model": load_config("configs/default.toml").model})
    elif args.mode == "blocks":
        config = config.model_copy(
            update={"model": config.model.model_copy(update={"nested_execution": "blocks"})}
        )
    train = config.train
    size = config.data.crop_size
    clean = jax.device_put(
        np.fromfile(args.batch_file, np.float32).reshape(train.effective_batch_size, size, size, 1)
    )
    full = create_model(config.model, rngs=nnx.Rngs(train.initialization_seed))
    models = (
        [full.extract(i) for i in range(full.num_subnets)]
        if args.mode == "separate"
        else [full.extract(0) if args.mode == "core" else full]
    )
    optimizers = [create_optimizer(model, config)[0] for model in models]
    emas = [jax.tree.map(lambda x: x.copy(), nnx.state(m, nnx.Param)) for m in models]
    gradient, update, _ = make_steps(config, all_subnets=args.mode != "separate")
    prepare = jax.jit(
        lambda x, k: training_inputs(x, k, config.diffusion, config.matching, levels=args.levels)
    )
    measurements = []
    losses = []
    for step in range(1 + args.warmup + args.measure):
        started = time.perf_counter()
        prepared = prepare(clean, jax.random.fold_in(jax.random.PRNGKey(train.seed), step))
        step_losses = []
        for index, (model, optimizer) in enumerate(zip(models, optimizers, strict=True)):
            metrics, accumulated, norm = accumulate_gradients(gradient, model, prepared, train)
            norm = float(norm)
            if not np.isfinite(norm):
                raise FloatingPointError("Nonfinite gradients")
            _, emas[index] = update(model, optimizer, accumulated, emas[index])
            jax.block_until_ready((nnx.state(model), nnx.state(optimizer), emas[index]))
            step_losses.append(float(metrics["loss"]))
        elapsed = time.perf_counter() - started
        measurements.append(elapsed)
        losses.append(step_losses)
        print(json.dumps({"step": step, "seconds": elapsed, "loss": step_losses}), flush=True)
    # Warm inference separately from the measured updates.
    inference = nnx.jit(recursive_forward_all)
    one = jax.tree.map(lambda x: x[:1], prepared)
    inference_times = []
    for repeat in range(1 + args.measure):
        started = time.perf_counter()
        predictions = [inference(m, one.noisy, one.noise_levels) for m in models]
        jax.block_until_ready(predictions)
        if repeat:
            inference_times.append(time.perf_counter() - started)
    measured = measurements[1 + args.warmup :]
    report = {
        "mode": args.mode,
        "accumulation": "jitted tree sums, averages and norm",
        "levels": args.levels,
        "devices": [str(x) for x in jax.devices()],
        "parameters": [parameter_count(m) for m in models],
        "config": config.model_dump(mode="json"),
        "first_update_seconds": measurements[0],
        "warmup_seconds": measurements[1 : 1 + args.warmup],
        "update_seconds": measured,
        "mean_update_seconds": statistics.mean(measured),
        "median_update_seconds": statistics.median(measured),
        "batch1_inference_seconds": inference_times,
        "median_batch1_inference_seconds": statistics.median(inference_times),
        "losses": losses,
    }
    from pathlib import Path

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
