"""Choose a fixed auxiliary coefficient from gradients, without fitting/evaluation feedback."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import jax
import numpy as np
from flax import nnx

from lineartgen.config import load_config
from lineartgen.data.geometry import mix_geometry
from lineartgen.data.pipeline import batches, discover, split_experiment
from lineartgen.diffusion import training_inputs
from lineartgen.models.serialization import load_inference
from lineartgen.training.steps import accumulate_gradients, make_steps


def flattened(tree):
    return np.concatenate([np.asarray(x).reshape(-1) for x in jax.tree.leaves(tree)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--auxiliary", choices=["boundary", "centerline"], default="boundary")
    args = parser.parse_args()
    field = "boundary_weight" if args.auxiliary == "boundary" else "cldice_weight"
    assert not args.output.exists(), "Do not overwrite a completed calibration"
    config = load_config(args.config)
    model, saved = load_inference(args.initialize_from)
    assert saved.model == config.model and getattr(config.loss, field) == 1
    initial = flattened(nnx.state(model, nnx.Param)).copy()
    base = config.model_copy(update={"loss": config.loss.model_copy(update={field: 0.0})})
    gradients = [make_steps(c)[0] for c in (base, config)]
    calibration_seed = config.train.seed + 1000
    train, _, _ = split_experiment(discover(config.data), config.data, config.train.seed)
    prepare = jax.jit(
        lambda x, key, levels: training_inputs(
            x, key, config.diffusion, config.matching, levels=levels, sketch=config.sketch
        ),
        static_argnums=(2,),
    )
    records = []
    stream = batches(train, config.data, batch_size=16, count=12, seed=calibration_seed)
    try:
        for step in range(12):
            started = time.perf_counter()
            batch = next(stream)
            clean, generated = mix_geometry(
                batch.images,
                config.geometry,
                seed=calibration_seed,
                step=step,
                image_scale=config.data.image_scale,
            )
            # Two independent training batches at each depth, without using development cases.
            levels = step % 6 + 1
            prepared = prepare(
                jax.device_put(clean),
                jax.random.fold_in(jax.random.PRNGKey(calibration_seed), step),
                levels,
            )
            parts = [accumulate_gradients(fn, model, prepared, config.train) for fn in gradients]
            g0, g1 = [flattened(p[1]) for p in parts]
            auxiliary = g1 - g0
            norm0, norm_aux = float(np.linalg.norm(g0)), float(np.linalg.norm(auxiliary))
            assert norm0 > 0 and norm_aux > 0 and np.isfinite(norm_aux)
            row = dict(
                batch=step,
                levels=levels,
                dataset_scale=batch.scale_level,
                generated=generated,
                base_loss=float(parts[0][0]["loss"]),
                auxiliary_loss=float(parts[1][0][args.auxiliary]),
                base_gradient_norm=norm0,
                auxiliary_gradient_norm=norm_aux,
                unweighted_ratio=norm_aux / norm0,
                cosine=float(np.dot(g0, auxiliary) / (norm0 * norm_aux)),
                seconds=time.perf_counter() - started,
            )
            assert all(not isinstance(x, float) or np.isfinite(x) for x in row.values())
            records.append(row)
            print(json.dumps(row), flush=True)
    finally:
        stream.close()
    np.testing.assert_array_equal(initial, flattened(nnx.state(model, nnx.Param)))
    ratios = np.array([r["unweighted_ratio"] for r in records])
    # Aim for 10% median contribution, capped at 50% on every calibration batch.
    exact = min(0.1 / np.median(ratios), 0.5 / ratios.max())
    coefficient = float(f"{exact:.6g}")
    result = dict(
        auxiliary=args.auxiliary,
        coefficient=coefficient,
        selection="min(.10 / median(unweighted_ratio), .50 / max(unweighted_ratio)); "
        "six significant digits",
        coefficient_frozen_before_training=True,
        calibration_seed=calibration_seed,
        calibration_batches=12,
        batch_size=16,
        records=records,
        achieved_median_ratio=float(np.median(ratios) * coefficient),
        achieved_max_ratio=float(ratios.max() * coefficient),
        weights_unchanged=True,
        reference=str(args.initialize_from.resolve()),
        reference_weights_sha256=hashlib.sha256(
            (args.initialize_from / "weights.npz").read_bytes()
        ).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    source = args.config.read_text()
    placeholder = f"{field} = 1.0 # Placeholder; calibration must replace before training."
    assert placeholder in source
    args.config.write_text(
        source.replace(
            placeholder,
            f"{field} = {coefficient} # Fixed by training-only gradient calibration.",
        )
    )
    print(f"Fixed coefficient: {coefficient}", flush=True)


if __name__ == "__main__":
    main()
