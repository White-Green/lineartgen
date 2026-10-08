"""Paired CPU latency through the actual BGRA API; no quality selection."""

import argparse
import hashlib
import os
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from campaign_protocol import read, sha, write
from lineartgen_runtime import LineartModel
from lineartgen_runtime.images import pyramid_plan


def benchmark(root):
    plan = read(root / "plan.json")
    if (root / "results.json").exists():
        raise FileExistsError(root / "results.json")
    runtimes, models = {}, {}
    for name, spec in plan["models"].items():
        checkpoint = Path(spec["checkpoint"])
        assert {n: sha(checkpoint / n) for n in spec["checkpoint_files"]} == spec[
            "checkpoint_files"
        ]
        model_path = Path(spec["onnx"])
        if spec.get("export", False):
            if model_path.exists():
                raise FileExistsError(model_path)
            from lineartgen.export import export_model
            from lineartgen.models.serialization import load_inference

            model, config = load_inference(checkpoint)
            export_model(model, config, model_path, provenance=str(checkpoint))
        else:
            assert sha(model_path) == spec["onnx_sha256"]
        runtimes[name] = LineartModel(model_path, num_threads=plan["threads"], provider="cpu")
        metadata = read(model_path.with_suffix(".json"))
        assert metadata["source_conditioning"] and metadata["model"]["output_mode"] == "residual"
        assert metadata["model"]["detail_channels"] == spec["detail_channels"]
        assert metadata["image_scale"] == plan["image_scale"]
        models[name] = dict(onnx_sha256=sha(model_path), metadata=metadata)
    keys = list(runtimes)
    assert len(keys) == 2
    records = []
    load_before = os.getloadavg()
    for height, width in plan["shapes_height_width"]:
        scribble = np.zeros((height, width, 4), np.uint8)
        scribble[:, width // 3 : width // 3 + 2, 3] = 255
        scribble[height // 2 - 3 : height // 2 + 3, :, 3] = 85
        scribble[:, width // 2 - 10 : width // 2 + 10, 3] = 25
        scribble[:, width // 2 - 3 : width // 2 + 3, 3] = 120
        lineart = np.zeros_like(scribble)
        lineart[:17, :, :3] = 255
        lineart[:17, :, 3] = 255
        lineart[height // 4 : height // 4 + 2, :, 3] = 255
        lineart[-23:, :, 3] = 64
        arguments = dict(
            scribble=scribble.tobytes(),
            lineart=lineart.tobytes(),
            width=width,
            height=height,
            strength=plan["strength"],
            seed=plan["seed"],
        )
        dimensions = pyramid_plan(height, width)
        for steps in plan["steps"]:
            timings = {name: [] for name in keys}
            hashes = {}
            for repeat in range(-plan["warmups"], plan["repeats"]):
                order = keys if repeat % 2 == 0 else list(reversed(keys))
                for name in order:
                    started = time.perf_counter_ns()
                    output = runtimes[name].infer(**arguments, denoise_steps=steps)
                    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
                    digest = hashlib.sha256(output).hexdigest()
                    assert len(output) == height * width * 4
                    if name in hashes:
                        assert digest == hashes[name]
                    hashes[name] = digest
                    if repeat >= 0:
                        timings[name].append(elapsed_ms)
            summaries = {
                name: dict(
                    median_ms=float(np.median(values)),
                    p95_ms=float(np.percentile(values, 95)),
                    minimum_ms=float(np.min(values)),
                )
                for name, values in timings.items()
            }
            record = dict(
                height=height,
                width=width,
                steps=steps,
                pyramid=dimensions.sizes,
                times_ms=timings,
                summary=summaries,
                output_sha256=hashes,
                median_ratio=summaries[keys[1]]["median_ms"] / summaries[keys[0]]["median_ms"],
            )
            records.append(record)
            write(root / "progress.json", dict(completed=len(records), latest=record))
            print(dict(shape=[height, width], steps=steps, summary=summaries), flush=True)
    cpu = next(
        x.split(":", 1)[1].strip()
        for x in Path("/proc/cpuinfo").read_text().splitlines()
        if x.startswith("model name")
    )
    write(
        root / "results.json",
        dict(
            plan_sha256=sha(root / "plan.json"),
            script_sha256=sha(__file__),
            runtime_files={
                str(p): sha(p)
                for p in Path("packages/lineartgen-runtime/src/lineartgen_runtime").glob("*.py")
            },
            models=models,
            records=records,
            environment=dict(
                cpu=cpu,
                platform=platform.platform(),
                python=platform.python_version(),
                numpy=np.__version__,
                onnxruntime=ort.__version__,
                provider="CPUExecutionProvider",
                threads=plan["threads"],
                load_before=load_before,
                load_after=os.getloadavg(),
            ),
            limitations=plan["limitations"],
            adoption=False,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    benchmark(p.parse_args().root.resolve())
