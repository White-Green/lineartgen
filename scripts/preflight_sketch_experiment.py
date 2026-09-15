"""Exercise the real batch, every recursion depth and all sketch variants on the training device."""

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np
from flax import nnx

from lineartgen.config import load_config
from lineartgen.data.geometry import mix_geometry
from lineartgen.diffusion import training_inputs
from lineartgen.models.serialization import load_inference
from lineartgen.training.comparison import comparison_configs
from lineartgen.training.steps import accumulate_gradients, create_optimizer, make_steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sketch.toml"))
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--study",
        choices=[
            "sketch",
            "sketch_rates",
            "geometry_data",
            "boundary_loss",
            "boundary_half",
            "cldice_loss",
        ],
        default="sketch",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    variants = comparison_configs(config, study=args.study)
    initial, initial_config = load_inference(args.initialize_from)
    assert initial_config.model == config.model
    with np.load(args.case, allow_pickle=False) as arrays:
        clean = jax.device_put(
            np.repeat(arrays["image"][None], config.train.effective_batch_size, axis=0)
        )
    assert clean.shape == (16, 512, 512, 1)
    numerical_steps = {name: make_steps(c) for name, c in variants.items()}
    models, preparations = {}, {}
    for name, settings in variants.items():
        model = nnx.clone(initial)
        optimizer, _ = create_optimizer(model, settings)
        ema = jax.tree.map(lambda x: x.copy(), nnx.state(model, nnx.Param))
        models[name] = model, optimizer, ema
        preparations[name] = jax.jit(
            lambda x, key, levels, c=settings: training_inputs(
                x, key, c.diffusion, c.matching, levels=levels, sketch=c.sketch
            ),
            static_argnums=(2,),
        )
    records = []
    for levels in range(1, 7):
        reference = None
        for name, settings in variants.items():
            started = time.perf_counter()
            source, generated = mix_geometry(
                np.asarray(clean),
                settings.geometry,
                seed=settings.train.seed,
                step=levels,
                image_scale=settings.data.image_scale,
            )
            prepared = preparations[name](
                jax.device_put(source), jax.random.PRNGKey(0xAB00 + levels), levels
            )
            jax.block_until_ready(prepared)
            prep_seconds = time.perf_counter() - started
            if reference is not None:
                jax.tree.map(np.testing.assert_array_equal, reference, prepared.noise_levels)
            reference = prepared.noise_levels
            for source, target, mask in zip(
                prepared.noisy, prepared.targets, prepared.noise_levels, strict=True
            ):
                protected = np.asarray(mask) == 0
                np.testing.assert_array_equal(
                    np.asarray(source)[protected], np.asarray(target)[protected]
                )
            model, optimizer, ema = models[name]
            gradient, update, _ = numerical_steps[name]
            metrics, gradients, norm = accumulate_gradients(
                gradient, model, prepared, settings.train
            )
            assert np.isfinite(float(norm)) and float(norm) > 0
            assert all(np.isfinite(float(v)) for v in metrics.values())
            _, ema = update(model, optimizer, gradients, ema)
            jax.block_until_ready((nnx.state(model), ema))
            models[name] = model, optimizer, ema
            record = dict(
                method=name,
                generated=generated,
                levels=levels,
                loss=float(metrics["loss"]),
                gradient_norm=float(norm),
                preparation_seconds=prep_seconds,
                total_seconds=time.perf_counter() - started,
            )
            records.append(record)
            print(json.dumps(record), flush=True)
    args.output.write_text(
        json.dumps(
            dict(
                devices=[str(x) for x in jax.devices()],
                records=records,
                batch_size=16,
                shared_masks=True,
                protected_teacher_exact=True,
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
