"""Real-shape GPU updates of every arm at every recursion depth; no campaign weights modified."""

import argparse
import time
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, write
from flax import nnx

from lineartgen.config import load_config
from lineartgen.data.geometry import mix_geometry
from lineartgen.data.pipeline import batches, discover, split_experiment
from lineartgen.diffusion import training_inputs
from lineartgen.models.serialization import load_inference
from lineartgen.training import checkpoints, lineart_state
from lineartgen.training.lineart import make_profiles
from lineartgen.training.lineart_adversarial import patch_coordinates
from lineartgen.training.lineart_study import create_arm, study_methods, update_arm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "preflight"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    config = load_config(root / "config.json")
    protocol = read(root / "protocol.json")
    coefficients = read(root / "calibration.json")["coefficients"]
    initial, _ = load_inference(protocol["initial"])

    def create():
        return {
            n: create_arm(
                load_config(root / "configs" / f"{n}.json"),
                initial,
                adversarial=n == "patch_prior",
                patch_size=protocol["patch_size"],
            )
            for n in study_methods(protocol)
        }

    arms = create()
    for arm in arms.values():
        jax.tree.map(np.testing.assert_array_equal, nnx.state(initial), nnx.state(arm["model"]))
    training, _, _ = split_experiment(discover(config.data), config.data, config.train.seed)
    stream = batches(
        training, config.data, batch_size=config.train.effective_batch_size, count=6, seed=20263006
    )
    prepare = jax.jit(
        lambda x, k, levels: training_inputs(
            x, k, config.diffusion, config.matching, levels=levels, sketch=config.sketch
        ),
        static_argnums=2,
    )
    rows = []
    for step, batch in enumerate(stream):
        clean, records = mix_geometry(
            batch.images,
            config.geometry,
            seed=20263006,
            step=step,
            image_scale=config.data.image_scale,
            include_paths=True,
        )
        profiles, coordinates = jax.device_put(
            (
                make_profiles(records, len(clean), len(clean[0]), seed=20263006, step=step),
                patch_coordinates(clean, seed=20263006, step=step, size=protocol["patch_size"]),
            )
        )
        prepared = prepare(jax.device_put(clean), jax.random.PRNGKey(step), step + 1)
        for name, arm in arms.items():
            started = time.perf_counter()
            metrics = update_arm(
                arm,
                prepared,
                profiles,
                coordinates,
                prior_weight=coefficients.get("patch_prior", 0.0),
            )
            record = dict(
                method=name, levels=step + 1, seconds=time.perf_counter() - started, **metrics
            )
            print(record, flush=True)
            rows.append(record)
    with checkpoints.manager(output / "checkpoints", keep=1) as saved:
        lineart_state.save(saved, arms, 6)
        restored = create()
        assert lineart_state.restore(saved, restored) == 6
        jax.tree.map(
            np.testing.assert_array_equal,
            lineart_state.payload(arms, 6),
            lineart_state.payload(restored, 6),
        )
    write(
        output / "results.json",
        dict(
            devices=[str(d) for d in jax.devices()],
            updates=rows,
            common_initial_weights_exact=True,
            checkpoint_all_states_exact=True,
            comparison_checkpoint_modified=False,
        ),
    )


if __name__ == "__main__":
    main()
