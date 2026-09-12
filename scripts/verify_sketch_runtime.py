"""Exercise each trained S export through the unchanged Krita-facing BGRA API."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from lineartgen_runtime import LineartModel
from lineartgen_runtime.images import prepare_input, pyramid_plan, to_bgra


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=["control", "tone", "blur", "strokes"])
    args = parser.parse_args()
    root = args.root.resolve()
    records = []
    for method in args.methods:
        for mode in ("raw", "ema"):
            path = root / "exports/step-10000" / f"{method}-{mode}.onnx"
            model = LineartModel(path, provider="cpu")
            for width, height in ((33, 17), (512, 512)):
                scribble = np.zeros((height, width, 4), np.uint8)
                scribble[:, width // 2 : width // 2 + 2, 3] = 160
                lineart = np.zeros_like(scribble)
                # Opaque gray pixels exercise protected tones, not only black and white.
                lineart[: height // 3, : width // 3, :3] = 200
                lineart[: height // 3, : width // 3, 3] = 255
                plan = pyramid_plan(
                    height, width, model._metadata["minimum_size"], model._metadata["stop_size"]
                )
                options = dict(
                    scribble=scribble.tobytes(),
                    lineart=lineart.tobytes(),
                    width=width,
                    height=height,
                    seed=42,
                )
                clean, strength = prepare_input(
                    options["scribble"],
                    options["lineart"],
                    plan,
                    0.5,
                    model._metadata["image_scale"],
                )
                expected = to_bgra(clean, plan)
                expected_pixels = np.frombuffer(expected, np.uint8).reshape(height, width, 4)
                protected = strength[0, :height, :width, 0] == 0
                for steps in (1, 5):
                    identity = model.infer(**options, strength=0, denoise_steps=steps)
                    assert identity == expected
                    actual = model.infer(**options, strength=0.5, denoise_steps=steps)
                    assert actual == model.infer(**options, strength=0.5, denoise_steps=steps)
                    assert len(actual) == width * height * 4 and actual != identity
                    pixels = np.frombuffer(actual, np.uint8).reshape(height, width, 4)
                    assert not pixels[..., :3].any()
                    np.testing.assert_array_equal(pixels[protected], expected_pixels[protected])
                    records.append(
                        dict(
                            method=method,
                            mode=mode,
                            width=width,
                            height=height,
                            denoise_steps=steps,
                            recursion_levels=len(plan.sizes),
                            identity_exact=True,
                            deterministic=True,
                            protected_bgra_exact=True,
                            output_sha256=hashlib.sha256(actual).hexdigest(),
                            model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        )
                    )
            print(f"Verified BGRA API: {method} {mode}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                checkpoint_step=10000,
                provider="CPUExecutionProvider",
                records=records,
                note="Preservation is relative to existing prepare_input/to_bgra behavior, "
                "including image_scale, quantization and alpha threshold. "
                "This is an API check, not a quality benchmark.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
