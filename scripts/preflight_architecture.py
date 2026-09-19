"""GPU checks for the architecture campaign, with a replay of the old default."""

import argparse
import subprocess
import time
import types

import jax
import numpy as np
from campaign_protocol import write
from flax import nnx

from lineartgen.config import ModelConfig, load_config
from lineartgen.data.geometry import mix_geometry
from lineartgen.data.pipeline import batches, discover, split_experiment
from lineartgen.diffusion import recursive_forward_all, training_inputs
from lineartgen.models import create_model
from lineartgen.training.steps import accumulate_gradients, create_optimizer, make_steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    train, _, _ = split_experiment(discover(config.data), config.data, config.train.seed)
    stream = batches(
        train,
        config.data,
        batch_size=config.train.effective_batch_size,
        count=1,
        seed=config.train.seed,
    )
    try:
        batch = next(stream)
    finally:
        stream.close()
    images, generated = mix_geometry(
        batch.images,
        config.geometry,
        seed=config.train.seed,
        step=0,
        image_scale=config.data.image_scale,
    )
    shapes = {
        f"{skip}_{d}": config.model.model_copy(update={"skip_mode": skip, "bottleneck_dilation": d})
        for skip in ("post_pool", "pre_pool")
        for d in (1, 2)
    }
    shapes["core"] = shapes["pre_pool_2"].model_copy(update={"nested_channels": ()})
    shapes["large"] = ModelConfig(
        channels=(4, 10, 20, 40),
        convolutions_per_block=3,
        output_mode="residual",
        skip_mode="pre_pool",
        bottleneck_dilation=2,
    )
    models = {name: create_model(c, rngs=nnx.Rngs(42)) for name, c in shapes.items()}
    for name in ("post_pool_2", "pre_pool_1", "pre_pool_2"):
        jax.tree.map(
            np.testing.assert_array_equal,
            nnx.state(models["post_pool_1"], nnx.Param),
            nnx.state(models[name], nnx.Param),
        )
    # Run the exact pre-change class on the same arrays; its weights are initialized identically.
    legacy = types.ModuleType("lineartgen.models._campaign_legacy")
    legacy.__package__ = "lineartgen.models"
    source = subprocess.check_output(
        ["git", "show", "cd3b415c8ad9f56ab058cd2086822e8a071c1eaa:src/lineartgen/models/nested.py"],
        text=True,
    )
    exec(compile(source, "legacy_nested.py", "exec"), legacy.__dict__)
    old = legacy.NestedUNet(config.model, rngs=nnx.Rngs(42))
    forward = nnx.jit(recursive_forward_all)
    prepare = jax.jit(
        lambda x, k, levels: training_inputs(
            x, k, config.diffusion, config.matching, levels=levels, sketch=config.sketch
        ),
        static_argnums=(2,),
    )
    prepared = {levels: prepare(images, jax.random.PRNGKey(47), levels) for levels in range(1, 7)}
    for data in prepared.values():
        a, b = (
            forward(old, data.noisy, data.noise_levels),
            forward(models["post_pool_1"], data.noisy, data.noise_levels),
        )
        jax.tree.map(np.testing.assert_array_equal, a, b)
    records = []
    for name, model in models.items():
        settings = config.model_copy(update={"model": shapes[name]})
        gradient, update, _ = make_steps(settings)
        optimizer, _ = create_optimizer(model, settings)
        ema = jax.tree.map(lambda x: x.copy(), nnx.state(model, nnx.Param))
        parameters = [
            sum(
                x.size
                for x in jax.tree.leaves(
                    nnx.state(model.extract(i) if hasattr(model, "extract") else model, nnx.Param)
                )
            )
            for i in range(getattr(model, "num_subnets", 1))
        ]
        for levels, data in prepared.items():
            start = time.perf_counter()
            metrics, gradients, norm = accumulate_gradients(gradient, model, data, settings.train)
            assert np.isfinite(float(norm)) and np.isfinite(float(metrics["loss"]))
            _, ema = update(model, optimizer, gradients, ema)
            jax.block_until_ready(ema)
            record = dict(
                architecture=name,
                levels=levels,
                parameters=parameters,
                loss=float(metrics["loss"]),
                gradient_norm=float(norm),
                seconds=time.perf_counter() - start,
            )
            records.append(record)
            print(record, flush=True)
    write(
        args.output,
        dict(
            devices=[str(d) for d in jax.devices()],
            records=records,
            original_default_recursion_bit_exact=True,
            paired_initial_parameters_exact=True,
            generated_examples=len(generated),
        ),
    )


if __name__ == "__main__":
    main()
