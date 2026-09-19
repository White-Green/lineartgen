"""Verify every trained prefix; benchmark and check deployable scalar-insert exports."""

import argparse
import hashlib
import time
from pathlib import Path

import jax
import numpy as np
from analyze_completion_campaign import check_seal, parameter_count
from campaign_protocol import read, sha, write
from flax import nnx
from lineartgen_runtime import LineartModel
from lineartgen_runtime.images import prepare_input, pyramid_plan, to_bgra
from lineartgen_runtime.sampling import denoise
from verify_training_methods import checked_checkpoint

from lineartgen.diffusion import (
    recursive_forward,
    recursive_forward_all,
    sample_prepared,
    training_inputs,
)
from lineartgen.evaluation_sampling import runtime_inputs
from lineartgen.export import export_model
from lineartgen.models.serialization import load_inference, save_inference
from lineartgen.training.steps import accumulate_gradients, create_optimizer, make_steps


def case_inputs(root, index=167, levels=3):
    case = read(root / "development/cases.json")["cases"][index]
    with np.load(root / "development" / case["file"], allow_pickle=False) as data:
        x = (data["source"] if "source" in data else data["image"])[None]
        t = data["strength"][None]
    return x, t, case, runtime_inputs(x, t, case["seed"], levels)


def check_close(a, b):
    errors = []
    for left, right in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        left, right = np.asarray(left), np.asarray(right)
        assert np.isfinite(left).all() and np.isfinite(right).all()
        np.testing.assert_allclose(left, right, atol=2e-5, rtol=2e-4)
        errors.append(float(np.abs(left - right).max()))
    return max(errors)


def bgra_checks(runtime):
    records = []
    for width, height in ((33, 17), (512, 512)):
        scribble = np.zeros((height, width, 4), np.uint8)
        scribble[:, width // 2 : width // 2 + 2, 3] = 160
        lineart = np.zeros_like(scribble)
        lineart[: height // 3, : width // 3, :3] = 200
        lineart[: height // 3, : width // 3, 3] = 255
        plan = pyramid_plan(
            height, width, runtime._metadata["minimum_size"], runtime._metadata["stop_size"]
        )
        options = dict(
            scribble=scribble.tobytes(),
            lineart=lineart.tobytes(),
            width=width,
            height=height,
            seed=42,
        )
        clean, strength = prepare_input(
            options["scribble"], options["lineart"], plan, 0.5, runtime._metadata["image_scale"]
        )
        expected = to_bgra(clean, plan)
        protected = strength[0, :height, :width, 0] == 0
        expected_pixels = np.frombuffer(expected, np.uint8).reshape(height, width, 4)
        for steps in (1, 5):
            identity = runtime.infer(**options, strength=0.0, denoise_steps=steps)
            assert identity == expected
            output = runtime.infer(**options, strength=0.5, denoise_steps=steps)
            assert output == runtime.infer(**options, strength=0.5, denoise_steps=steps)
            pixels = np.frombuffer(output, np.uint8).reshape(height, width, 4)
            np.testing.assert_array_equal(pixels[protected], expected_pixels[protected])
            records.append(
                dict(
                    width=width,
                    height=height,
                    denoise_steps=steps,
                    identity_exact=True,
                    protected_exact=True,
                    deterministic=True,
                    output_sha256=hashlib.sha256(output).hexdigest(),
                )
            )
    return records


def benchmark(model, config, root):
    inference = nnx.jit(lambda m, x, t: sample_prepared(m, x, t, steps=1))
    prepared = case_inputs(root)[3]
    infer = []
    for index in range(getattr(model, "num_subnets", 1)):
        prefix = model.extract(index) if hasattr(model, "extract") else model
        times = []
        for _repeat in range(31):
            start = time.perf_counter()
            out = inference(prefix, *prepared)
            jax.block_until_ready(out)
            times.append(time.perf_counter() - start)
        infer.append(
            dict(
                subnet=index,
                first_seconds=times[0],
                median_seconds=float(np.median(times[1:])),
                seconds=times[1:],
                parameters=sum(x.size for x in jax.tree.leaves(nnx.state(prefix, nnx.Param))),
            )
        )
    updates = []
    x = case_inputs(root)[0]
    batch = np.repeat(x, config.train.effective_batch_size, axis=0)
    for levels in (1, 6):
        copied = nnx.clone(model)
        gradient, update, _ = make_steps(config)
        optimizer, _ = create_optimizer(copied, config)
        ema = jax.tree.map(lambda x: x.copy(), nnx.state(copied, nnx.Param))
        data = training_inputs(
            batch,
            jax.random.PRNGKey(47),
            config.diffusion,
            config.matching,
            levels=levels,
            sketch=config.sketch,
        )
        jax.block_until_ready(data)
        times = []
        for _repeat in range(16):
            start = time.perf_counter()
            metrics, grads, norm = accumulate_gradients(gradient, copied, data, config.train)
            assert np.isfinite(float(norm)) and np.isfinite(float(metrics["loss"]))
            _, ema = update(copied, optimizer, grads, ema)
            jax.block_until_ready(ema)
            times.append(time.perf_counter() - start)
        updates.append(
            dict(
                levels=levels,
                first_seconds=times[0],
                warmup_seconds=times[1:6],
                median_seconds=float(np.median(times[6:])),
                seconds=times[6:],
            )
        )
    return dict(
        inference=infer,
        updates=updates,
        scope="GPU; inference at runtime depth 3. Update timing excludes I/O/preparation, "
        "repeats a fixed development input in batch16, and updates disposable clones.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    seal = check_seal(root)
    step = seal["step"]
    cohorts = read(root / "cohorts.json")
    records = []
    timings = {}
    forward_all = nnx.jit(recursive_forward_all)
    forward = nnx.jit(recursive_forward)
    sampler = nnx.jit(lambda m, x, t: sample_prepared(m, x, t, steps=1))
    for seed, cohort in cohorts.items():
        for role, name in cohort.items():
            arm = root / "arms" / name
            for mode in ("raw", "ema"):
                snapshot = arm / f"snapshots/step-{step}/model/{mode}"
                model, config = load_inference(snapshot)
                if args.export:
                    scalar = model.extract(0) if hasattr(model, "extract") else model
                    path = root / "exports" / f"{name}-{mode}.onnx"
                    if not path.exists():
                        export_model(scalar, config, path, provenance=f"{name} {mode} step {step}")
                    runtime = LineartModel(path, num_threads=1, provider="cpu")
                    errors = []
                    for index in (167, 256, 479, 1376):
                        x, t, case, prepared = case_inputs(root, index)
                        expected = sampler(scalar, *prepared)
                        actual = denoise(
                            runtime.forward, x, t, seed=case["seed"], steps=1, levels=3
                        )
                        errors.append(check_close(expected, actual))
                        np.testing.assert_array_equal(actual[t == 0], x[t == 0])
                    records.append(
                        dict(
                            arm=name,
                            mode=mode,
                            onnx=str(path),
                            sha256=sha(path),
                            runtime_max_error=max(errors),
                            bgra=bgra_checks(runtime),
                        )
                    )
                else:
                    restored = checked_checkpoint(
                        arm / "training/model", config, step, use_ema=mode == "ema"
                    )
                    jax.tree.map(
                        np.testing.assert_array_equal,
                        nnx.state(model, nnx.Param),
                        nnx.state(restored, nnx.Param),
                    )
                    comparisons = []
                    for levels in (1, 6):
                        _, _, _, prepared = case_inputs(root, levels=levels)
                        whole = forward_all(model, *prepared)
                        for index in range(getattr(model, "num_subnets", 1)):
                            prefix = model.extract(index) if hasattr(model, "extract") else model
                            count = sum(
                                x.size for x in jax.tree.leaves(nnx.state(prefix, nnx.Param))
                            )
                            assert count == parameter_count(
                                snapshot / "weights.npz",
                                index=index if hasattr(model, "extract") else None,
                            )
                            error = check_close(whole[index], forward(prefix, *prepared))
                            comparisons.append(dict(levels=levels, subnet=index, error=error))
                            if levels == 1:
                                destination = root / "extracted" / name / mode / str(index)
                                selected = config.model_copy(
                                    update={
                                        "model": config.model.model_copy(
                                            update={
                                                "nested_channels": config.model.nested_channels[
                                                    :index
                                                ]
                                            }
                                        )
                                    }
                                )
                                if not destination.exists():
                                    save_inference(
                                        prefix,
                                        selected,
                                        destination,
                                        provenance=f"{name} {mode} {step} subnet {index}",
                                    )
                                loaded, saved = load_inference(destination)
                                assert saved == selected
                                jax.tree.map(
                                    np.testing.assert_array_equal,
                                    nnx.state(prefix, nnx.Param),
                                    nnx.state(loaded, nnx.Param),
                                )
                                check_close(forward(prefix, *prepared), forward(loaded, *prepared))
                        if getattr(model, "num_subnets", 1) > 1:
                            # Change each extension independently; all smaller prefixes must
                            # remain bit exact through the entire recursion.
                            for extension in range(1, model.num_subnets):
                                changed = nnx.clone(model)
                                for block in (*changed.encoders, *changed.decoders):
                                    for layer in block.layers:
                                        layer.groups[extension].kernel[...] *= -1
                                        layer.groups[extension].bias[...] += 0.2
                                head = changed.heads[extension].layers[-1]
                                head.kernel[...] = jax.numpy.zeros_like(head.kernel[...])
                                head.bias[...] = -jax.numpy.ones_like(head.bias[...])
                                altered = forward_all(changed, *prepared)
                                jax.tree.map(
                                    np.testing.assert_array_equal,
                                    whole[:extension],
                                    altered[:extension],
                                )
                                assert all(np.isfinite(x).all() for x in jax.tree.leaves(altered))
                                assert any(
                                    not np.array_equal(a, b)
                                    for a, b in zip(
                                        whole[extension], altered[extension], strict=True
                                    )
                                )
                    records.append(
                        dict(arm=name, mode=mode, checkpoint_step=step, comparisons=comparisons)
                    )
                    if seed == "42" and mode == "ema":
                        timings[role] = benchmark(model, config, root)
                print(f"Checked {name} {mode}", flush=True)
    output = root / ("runtime-verification.json" if args.export else "model-verification.json")
    write(
        output,
        dict(
            records=records,
            timings=timings,
            devices=[str(d) for d in jax.devices()],
            export_scope="Shared S, independent S and dense baseline use the current "
            "scalar-insert ONNX/BGRA API. M/L are validated as autonomous JAX prefixes; "
            "their multi-prefix inserts are not supported by that ONNX API.",
        ),
    )
    print(output)


if __name__ == "__main__":
    main()
