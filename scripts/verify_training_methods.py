"""Verify trained prefixes and export their S models using actual operational inputs.

Run verification on the training GPU, then exports with JAX_PLATFORMS=cpu.
Snapshots are independent of the checkpoint manager's retention policy.
"""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from lineartgen.config import load_config
from lineartgen.diffusion import pyramid, recursive_forward, recursive_forward_all, sample
from lineartgen.export import export_model
from lineartgen.models import create_model
from lineartgen.models.serialization import load_inference, save_inference
from lineartgen.training import checkpoints
from lineartgen.training.comparison import comparison_configs
from lineartgen.training.steps import create_optimizer


def inputs(root, levels):
    # A line-containing crop at dataset scale 5, strength .7, with partial protection.
    manifest = json.loads((root / "evaluation/cases.json").read_text())
    case = manifest["cases"][167]
    assert case["scale_level"] == 5 and case["partial"] and case["strength"] == 0.7
    path = root / "evaluation" / case["file"]
    with np.load(path, allow_pickle=False) as arrays:
        clean, strength = jax.device_put((arrays["image"][None], arrays["strength"][None]))
    targets, noise_levels = pyramid(clean, levels), pyramid(strength, levels)
    key = jax.random.PRNGKey(case["seed"])
    keys = (key,) if levels == 1 else jax.random.split(key, levels)
    noisy = tuple(
        target * jnp.sqrt(1 - level) + jax.random.normal(k, target.shape) * jnp.sqrt(level)
        for target, level, k in zip(targets, noise_levels, keys, strict=True)
    )
    return clean, strength, noisy, noise_levels, key, hashlib.sha256(path.read_bytes()).hexdigest()


def max_error(left, right):
    errors = []
    for a, b in zip(left, right, strict=True):
        a, b = np.asarray(a), np.asarray(b)
        assert np.isfinite(a).all() and np.isfinite(b).all()
        np.testing.assert_allclose(a, b, atol=2e-5, rtol=2e-4)
        errors.append(float(np.abs(a - b).max()))
    return max(errors)


def checked_checkpoint(run, config, step, *, use_ema):
    assert load_config(run / "config.json") == config
    model = create_model(config.model, rngs=nnx.Rngs(config.train.initialization_seed))
    optimizer, _ = create_optimizer(model, config)
    ema = nnx.state(model, nnx.Param)
    with checkpoints.manager(run / "checkpoints", config.train.keep_checkpoints) as manager:
        # Supply a local-device template so a GPU checkpoint can also be audited on CPU.
        restored_step, ema = checkpoints.restore(manager, model, optimizer, ema, step)
    assert restored_step == step and int(optimizer.step[...]) == step
    if use_ema:
        nnx.update(model, ema)
    return model


def verify(root, step, modes, *, check_checkpoints=False, study="methods"):
    reference = json.loads((root / "training-source.json").read_text())
    for path, expected in reference["file_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    settings = comparison_configs(load_config(root / "config.toml"), study=study)
    forward_all, forward = nnx.jit(recursive_forward_all), nnx.jit(recursive_forward)
    sampler = nnx.jit(
        lambda m, x, k, t, levels: sample(m, x, k, strength=t, steps=5, levels=levels),
        static_argnums=(4,),
    )
    records = []
    for mode in modes:
        for method, config in settings.items():
            source = root / "snapshots" / f"step-{step}" / method / mode
            model, saved = load_inference(source)
            assert saved == config
            if check_checkpoints:
                run = root / "training" / method
                restored = checked_checkpoint(run, saved, step, use_ema=mode == "ema")
                jax.tree.map(
                    np.testing.assert_array_equal,
                    nnx.state(model, nnx.Param),
                    nnx.state(restored, nnx.Param),
                )
            counts, comparisons, archives = [], [], []
            for levels in (1, 6):
                clean, strength, noisy, noises, key, case_sha = inputs(root, levels)
                whole = forward_all(model, noisy, noises)
                for index in range(3):
                    prefix = model.extract(index)
                    error = max_error(whole[index], forward(prefix, noisy, noises))
                    comparisons.append(dict(levels=levels, subnet=index, max_abs_error=error))
                    if levels == 1:
                        destination = (
                            root / "extracted" / f"step-{step}" / method / mode / str(index)
                        )
                        subset = saved.model_copy(
                            update={
                                "model": saved.model.model_copy(
                                    update={"nested_channels": saved.model.nested_channels[:index]}
                                )
                            }
                        )
                        if not destination.exists():
                            save_inference(
                                prefix,
                                subset,
                                destination,
                                provenance=f"{method}: {mode} step {step}, subnet {index}",
                            )
                        restored, restored_config = load_inference(destination)
                        assert restored_config == subset
                        jax.tree.map(
                            np.testing.assert_array_equal,
                            nnx.state(prefix, nnx.Param),
                            nnx.state(restored, nnx.Param),
                        )
                        max_error(forward(prefix, noisy, noises), forward(restored, noisy, noises))
                        counts.append(
                            sum(x.size for x in jax.tree.leaves(nnx.state(prefix, nnx.Param)))
                        )
                        archives.append(
                            {
                                str(destination / name): hashlib.sha256(
                                    (destination / name).read_bytes()
                                ).hexdigest()
                                for name in ("config.json", "weights.npz", "metadata.json")
                            }
                        )
                changed = nnx.clone(model)
                for block in (*changed.encoders, *changed.decoders):
                    for layer in block.layers:
                        layer.groups[2].kernel[...] *= -1
                        layer.groups[2].bias[...] += 0.2
                changed.heads[2].layers[-1].kernel[...] = jnp.zeros_like(
                    changed.heads[2].layers[-1].kernel[...]
                )
                changed.heads[2].layers[-1].bias[...] = -jnp.ones_like(
                    changed.heads[2].layers[-1].bias[...]
                )
                altered = forward_all(changed, noisy, noises)
                assert all(np.isfinite(x).all() for x in jax.tree.leaves(altered))
                jax.tree.map(np.testing.assert_array_equal, whole[:2], altered[:2])
                assert any(
                    not np.array_equal(a, b) for a, b in zip(whole[2], altered[2], strict=True)
                )
                if config.model.output_mode == "residual":
                    protected = np.asarray(strength) == 0
                    for index in range(3):
                        actual = np.asarray(
                            sampler(model.extract(index), clean, key, strength, levels)
                        )
                        np.testing.assert_array_equal(
                            actual[protected], np.asarray(clean)[protected]
                        )
            assert counts == [5933, 10206, 17893]
            records.append(
                dict(
                    method=method,
                    mode=mode,
                    parameters=counts,
                    checkpoint_and_optimizer_step=step if check_checkpoints else None,
                    comparisons=comparisons,
                    archives=archives,
                    mutation="S/M unchanged at every scale while L changed",
                    archive_roundtrip="parameters exact; forward within tolerance",
                    protected_pixels="exact after five denoising steps"
                    if config.model.output_mode == "residual"
                    else "measured in operational evaluation",
                )
            )
            print(f"Verified trained {method} {mode} step {step}", flush=True)
    destination = root / "verification" / f"step-{step}-{'-'.join(modes)}.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(
        json.dumps(
            dict(
                step=step,
                devices=[str(x) for x in jax.devices()],
                case=167,
                case_sha256=case_sha,
                records=records,
            ),
            indent=2,
        )
        + "\n"
    )


def exports(root, step, modes, *, study="methods"):
    import onnxruntime as ort

    records = []
    settings = comparison_configs(load_config(root / "config.toml"), study=study)
    for mode in modes:
        for method in settings:
            model, config = load_inference(
                root / "extracted" / f"step-{step}" / method / mode / "0"
            )
            output = root / "exports" / f"step-{step}" / f"{method}-{mode}.onnx"
            export_model(model, config, output, provenance=f"{method}: subnet 0 {mode} step {step}")
            _, _, noisy, noises, _, case_sha = inputs(root, 6)
            expected = nnx.jit(recursive_forward)(model, noisy, noises)
            feeds = dict(
                image=np.asarray(noisy[0]),
                noise_level=np.asarray(noises[0]),
                insert=np.asarray(expected[1]),
            )
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            session = ort.InferenceSession(
                str(output), sess_options=options, providers=["CPUExecutionProvider"]
            )
            actual = session.run(["prediction"], feeds)[0]
            error = max_error((actual,), (expected[0],))
            if config.model.output_mode == "residual":
                protected = feeds["noise_level"] == 0
                np.testing.assert_array_equal(actual[protected], feeds["image"][protected])
            metadata = json.loads(output.with_suffix(".json").read_text())
            records.append(
                dict(
                    method=method,
                    mode=mode,
                    onnx=str(output),
                    sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                    dynamic_validation=metadata["validation"],
                    operational_512_max_abs_error=error,
                )
            )
            print(f"Exported and checked {method} {mode}, 512px max error={error:.3g}", flush=True)
    (root / "exports" / f"step-{step}" / "validation.json").write_text(
        json.dumps(
            dict(
                step=step,
                devices=[str(x) for x in jax.devices()],
                case=167,
                case_sha256=case_sha,
                records=records,
            ),
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--step", type=int, default=10000)
    parser.add_argument("--modes", choices=["raw", "ema"], nargs="+", default=["raw", "ema"])
    parser.add_argument("--exports-only", action="store_true")
    parser.add_argument("--check-checkpoints", action="store_true")
    parser.add_argument(
        "--study",
        choices=[
            "methods",
            "sketch",
            "sketch_rates",
            "geometry_data",
            "boundary_loss",
            "boundary_half",
            "cldice_loss",
            "noise_strength",
        ],
        default="methods",
    )
    args = parser.parse_args()
    if args.exports_only:
        exports(args.root, args.step, args.modes, study=args.study)
    else:
        verify(
            args.root,
            args.step,
            args.modes,
            check_checkpoints=args.check_checkpoints,
            study=args.study,
        )


if __name__ == "__main__":
    main()
