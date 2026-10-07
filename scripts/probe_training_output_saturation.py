"""Measure blocked direct output gradients on fixed training-only batches before fine-tuning."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from campaign_protocol import read, sha, write
from flax import nnx

from lineartgen.config import load_config
from lineartgen.data.geometry import mix_geometry
from lineartgen.data.pipeline import batches, discover, split_experiment
from lineartgen.diffusion import recursive_forward, training_inputs
from lineartgen.models import unet
from lineartgen.models.serialization import load_inference


def probe(root):
    plan, protocol = read(root / "diagnostic-plan.json"), read(root / "protocol.json")
    destination = root / "training-saturation.json"
    if destination.exists():
        raise FileExistsError(destination)
    config = load_config(root / "config.json")
    model, _ = load_inference(protocol["initial"])
    surrogate = nnx.clone(model)
    surrogate.output_clip_gradient = "straight_through"
    training, _, _ = split_experiment(discover(config.data), config.data, config.train.seed)
    seed, count = plan["training_saturation_seed"], plan["training_saturation_batches"]
    stream = batches(
        training, config.data, batch_size=config.train.effective_batch_size, count=count, seed=seed
    )
    prepare = jax.jit(
        lambda x, k, levels: training_inputs(
            x, k, config.diffusion, config.matching, levels=levels, sketch=config.sketch
        ),
        static_argnums=2,
    )
    original, captured = unet.denoised_output, []

    def observed(raw, image, noise, mode, gradient="hard"):
        captured.append(image + jnp.sqrt(noise) * raw if mode == "residual" else raw)
        return original(raw, image, noise, mode, gradient)

    unet.denoised_output = observed

    @nnx.jit
    def inspect(network, prepared):
        captured.clear()
        outputs = recursive_forward(network, prepared.noisy, prepared.noise_levels)
        return outputs, tuple(reversed(captured))

    rows = []
    try:
        for step, batch in enumerate(stream):
            clean, _ = mix_geometry(
                batch.images,
                config.geometry,
                seed=seed,
                step=step,
                image_scale=config.data.image_scale,
            )
            levels = plan["training_depths"][step]
            prepared = prepare(
                jax.device_put(clean), jax.random.fold_in(jax.random.PRNGKey(seed), step), levels
            )
            outputs, raws = inspect(model, prepared)
            candidate = inspect(surrogate, prepared)
            jax.tree.map(np.testing.assert_array_equal, (outputs, raws), candidate)
            for level, (output, raw, target, noise) in enumerate(
                zip(outputs, raws, prepared.targets, prepared.noise_levels, strict=True)
            ):
                p, z, y, t = map(np.asarray, (output, raw, target, noise))
                np.testing.assert_array_equal(p, np.clip(z, -1, 1))
                truth, ink = np.clip((1 - y) / 2, 0, 1), np.clip((1 - p) / 2, 0, 1)
                active = t > config.diffusion.min_noise
                core = active & (truth > 0.02)
                deficit = np.maximum(truth - ink, 0) * core
                rows.append(
                    dict(
                        batch=step,
                        recursion_levels=levels,
                        level=level,
                        editable_core_pixels=int(core.sum()),
                        white_saturated_core_pixels=int((core & (z > 1)).sum()),
                        ink_deficit_mass=float(deficit.sum()),
                        white_saturated_deficit_mass=float((deficit * (z > 1)).sum()),
                    )
                )
            print(dict(batch=step, recursion_levels=levels, equal_forward=True), flush=True)
    finally:
        stream.close()
        unet.denoised_output = original
    write(
        destination,
        dict(
            rows=rows,
            hard_and_surrogate_forward_exact=True,
            initial_weights_sha256=sha(Path(protocol["initial"]) / "weights.npz"),
            script_sha256=sha(Path(__file__)),
            diagnostic_plan_sha256=sha(root / "diagnostic-plan.json"),
            scope="12 fixed training-only batches; diagnosis, not quality validation",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    probe(args.root.resolve())
