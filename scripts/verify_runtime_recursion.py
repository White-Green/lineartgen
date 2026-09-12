"""Compare actual S weights in JAX with unchanged ONNX runtime at runtime depth."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from flax import nnx
from lineartgen_runtime import LineartModel
from lineartgen_runtime.sampling import denoise
from summarize_training_methods import sha

from lineartgen.diffusion import sample_prepared
from lineartgen.evaluation_sampling import runtime_inputs, runtime_levels
from lineartgen.models.serialization import load_inference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.cases / "cases.json").read_text())
    run = nnx.jit(lambda m, x, t: sample_prepared(m, x, t, steps=1))
    records = []
    for method in args.methods:
        for mode in ("raw", "ema"):
            source = args.models_root / "snapshots/step-10000" / method / mode
            model, config = load_inference(source)
            model = model.extract(0)
            path = args.models_root / "exports/step-10000" / f"{method}-{mode}.onnx"
            ort = LineartModel(path, num_threads=1, provider="cpu")
            for index in (167, 256, 479, 1376):
                case = manifest["cases"][index]
                with np.load(args.cases / case["file"], allow_pickle=False) as arrays:
                    x = (arrays["source"] if "source" in arrays else arrays["image"])[None]
                    t = arrays["strength"][None]
                levels = runtime_levels(*x.shape[1:3], config.model.minimum_size)
                prepared = runtime_inputs(x, t, case["seed"], levels)
                expected = np.asarray(run(model, *prepared))
                actual = denoise(ort.forward, x, t, seed=case["seed"], steps=1, levels=levels)
                assert np.isfinite(actual).all() and np.isfinite(expected).all()
                np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4)
                np.testing.assert_array_equal(actual[t == 0], x[t == 0])
                records.append(
                    dict(
                        method=method,
                        mode=mode,
                        case=index,
                        levels=levels,
                        max_abs_error=float(np.abs(actual - expected).max()),
                        protected_exact=True,
                        onnx_sha256=sha(path),
                        weights_sha256=sha(source / "weights.npz"),
                    )
                )
            print(f"Checked {method}/{mode}", flush=True)
    args.output.write_text(
        json.dumps(
            dict(
                records=records,
                manifest_sha256=sha(args.cases / "cases.json"),
                devices=[str(x) for x in jax.devices()],
                scope="Normalized arrays, actual S ONNX and unchanged runtime sampling; "
                "no BGRA conversion",
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
