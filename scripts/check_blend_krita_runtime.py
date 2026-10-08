"""Export an actual selected checkpoint and check the BGRA API against JAX on CPU."""

import argparse
from pathlib import Path

import numpy as np
from campaign_protocol import sha, write
from flax import nnx
from lineartgen_runtime import LineartModel
from lineartgen_runtime.images import prepare_input, pyramid_plan, to_bgra
from lineartgen_runtime.sampling import denoise

from lineartgen.evaluation_sampling import inference_with_source, runtime_inputs
from lineartgen.export import export_model
from lineartgen.models.serialization import load_inference


def check(checkpoint, destination):
    destination.mkdir(parents=True, exist_ok=False)
    original = {n: sha(checkpoint / n) for n in ("weights.npz", "config.json", "metadata.json")}
    model, config = load_inference(checkpoint)
    assert config.model.source_conditioning
    path = export_model(model, config, destination / "model.onnx", provenance=str(checkpoint))
    runtime = LineartModel(path, num_threads=1, provider="cpu")
    inference = nnx.jit(
        lambda m, x, t, source, steps: inference_with_source(m, x, t, source, steps=steps),
        static_argnums=4,
    )
    records = []
    for height, width in [(145, 129), (512, 512), (513, 385)]:
        scribble = np.zeros((height, width, 4), np.uint8)
        # Low-opacity overlapping strokes, opaque detail, and a broad gray fringe.
        scribble[:, width // 3 : width // 3 + 2, 3] = 255
        scribble[height // 2 - 3 : height // 2 + 3, :, 3] = 85
        scribble[:, width // 2 - 10 : width // 2 + 10, 3] = 25
        scribble[:, width // 2 - 3 : width // 2 + 3, 3] = 120
        lineart = np.zeros_like(scribble)
        lineart[:17, :, :3] = 255  # Fully protected white region.
        lineart[:17, :, 3] = 255
        lineart[height // 4 : height // 4 + 2, :, 3] = 255
        lineart[-23:, :, 3] = 64  # Partial protection, not a binary mask.
        scribble_bytes, lineart_bytes = scribble.tobytes(), lineart.tobytes()
        plan = pyramid_plan(height, width, config.model.minimum_size)
        for strength in [0.0, 0.1, 0.5]:
            source, noise = prepare_input(
                scribble_bytes, lineart_bytes, plan, strength, config.data.image_scale
            )
            prepared = runtime_inputs(source, noise, 42, len(plan.sizes))
            for steps in [1, 2, 5]:
                expected = np.asarray(inference(model, *prepared, source, steps))
                actual_float = denoise(
                    runtime.forward,
                    source,
                    noise,
                    seed=42,
                    steps=steps,
                    levels=len(plan.sizes),
                    source_conditioning=True,
                    preserve_zero_noise=True,
                )
                np.testing.assert_allclose(actual_float, expected, atol=2e-4, rtol=2e-4)
                protected_error = float(
                    np.max(np.abs(actual_float[noise == 0] - source[noise == 0]))
                )
                assert protected_error == 0.0
                actual_bytes = runtime.infer(
                    scribble=scribble_bytes,
                    lineart=lineart_bytes,
                    width=width,
                    height=height,
                    strength=strength,
                    seed=42,
                    denoise_steps=steps,
                )
                assert actual_bytes == to_bgra(actual_float, plan)
                actual = np.frombuffer(actual_bytes, np.uint8).astype(int)
                wanted = np.frombuffer(to_bgra(expected, plan), np.uint8).astype(int)
                diff = np.abs(actual - wanted)
                # Quantization/alpha threshold can expose a sub-ULP backend difference;
                # record exact byte differences as well as continuous predictions.
                records.append(
                    dict(
                        height=height,
                        width=width,
                        levels=len(plan.sizes),
                        strength=strength,
                        denoise_steps=steps,
                        float_max_error=float(np.max(np.abs(actual_float - expected))),
                        byte_max_error=int(diff.max()),
                        bytes_differing=int(np.count_nonzero(diff)),
                        output_length_exact=len(actual_bytes) == height * width * 4,
                        protected_float_error=protected_error,
                    )
                )
        print(dict(shape=[height, width], completed=len(records)), flush=True)
    assert original == {n: sha(checkpoint / n) for n in original}
    write(
        destination / "runtime-check.json",
        dict(
            checkpoint=str(checkpoint),
            checkpoint_files=original,
            onnx_sha256=sha(path),
            script_sha256=sha(Path(__file__)),
            records=records,
            max_byte_error=max(r["byte_max_error"] for r in records),
            all_continuous_comparisons_passed=True,
            limitations=(
                "CPU ONNX Runtime versus CPU JAX; Windows DirectML and Krita GUI not exercised. "
                "Residual zero-noise masks must remain exactly unchanged."
            ),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    check(args.checkpoint.resolve(), args.destination.resolve())
