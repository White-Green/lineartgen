"""Dedicated fixed blank-paper diagnostic, separate from the reused development suite."""

import argparse
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from evaluate_lineart_loss_study import render
from flax import nnx
from lineartgen_runtime.images import pyramid_plan, to_bgra
from train_lineart_loss_study import verify

from lineartgen.diffusion import sample_prepared
from lineartgen.evaluation_sampling import runtime_inputs
from lineartgen.models.serialization import load_inference
from lineartgen.training.lineart_study import METHODS

SEEDS = (2026100609, 2026100610, 2026100611, 2026100612)
STRENGTHS = (0.0, 0.1, 0.3, 0.5, 0.7)


def evaluate(root, step, mode):
    protocol = verify(root)
    folder = root / "blank-evaluation" / f"step-{step}-{mode}"
    folder.mkdir(parents=True, exist_ok=True)
    paths = (
        {"initial": Path(protocol["initial"])}
        if step == 0
        else {name: root / "snapshots" / f"step-{step}" / name / mode for name in METHODS}
    )
    hashes = {name: sha(path / "weights.npz") for name, path in paths.items()}
    source_hash = sha(Path(__file__))
    if (folder / "metrics.json").exists():
        previous = read(folder / "metrics.json")
        assert previous["model_weights_sha256"] == hashes
        assert previous["analysis_source_sha256"] == source_hash
        return
    models = {name: load_inference(path)[0] for name, path in paths.items()}
    inference = nnx.jit(lambda m, x, t, s: sample_prepared(m, x, t, steps=s), static_argnums=3)
    white = np.ones((512, 512, 1), np.float32)
    rows, index = [], 0
    for seed in SEEDS:
        for strength in STRENGTHS:
            for partial in (False, True):
                noise = np.full_like(white, strength)
                if partial:
                    noise[:] = 0
                    noise[128:384, 128:384] = strength
                active = noise[..., 0] > 0
                prepared = runtime_inputs(white[None], noise[None], seed, levels=3)
                noisy = prepared[0][0][0]
                prepared = jax.device_put(prepared)
                for iterations in (1, 2, 5):
                    predictions = {}
                    for name, model in models.items():
                        prediction = np.asarray(inference(model, *prepared, iterations))[0]
                        assert np.isfinite(prediction).all()
                        error = np.abs(prediction[..., 0] - 1)
                        protected = float(error[~active].max()) if (~active).any() else None
                        assert protected in (None, 0.0)
                        opacity = np.clip((1 - prediction[..., 0]) / 2, 0, 1)
                        alpha = np.frombuffer(
                            to_bgra(prediction[None], pyramid_plan(512, 512)), np.uint8
                        ).reshape(512, 512, 4)[..., 3]
                        rows.append(
                            dict(
                                case=index,
                                seed=seed,
                                strength=strength,
                                partial=partial,
                                denoise_steps=iterations,
                                method=name,
                                active_mean_opacity=float(opacity[active].mean())
                                if active.any()
                                else None,
                                active_visible_fraction=float((alpha[active] > 0).mean())
                                if active.any()
                                else None,
                                maximum_opacity=float(opacity.max()),
                                rendered_ink_pixels=int((alpha > 0).sum()),
                                protected_max_error=protected,
                            )
                        )
                        predictions[name] = prediction
                    if seed == SEEDS[0] and strength == 0.5:
                        render(folder, index, iterations, white, white, noisy, predictions)
                index += 1
        print(f"Blank evaluation {index}/40 cases", flush=True)
    assert len(rows) == 40 * 3 * len(models)
    write(
        folder / "metrics.json",
        dict(
            step=step,
            mode=mode,
            case_count=40,
            prediction_count=len(rows),
            results=rows,
            model_weights_sha256=hashes,
            analysis_source_sha256=source_hash,
            seeds=SEEDS,
            strengths=STRENGTHS,
            recursion_levels=3,
            size=512,
            input="Clean white paper plus independent runtime NumPy noise, whole/central edit",
            limitation="Dedicated synthetic sanity check, not evidence of sketch reconstruction",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--mode", choices=("ema", "raw"), default="ema")
    args = parser.parse_args()
    evaluate(args.root.resolve(), args.step, args.mode)


if __name__ == "__main__":
    main()
