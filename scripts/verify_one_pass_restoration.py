"""Actual trained ONNX/BGRA contract for an explicitly labeled restoration mode."""

import argparse
import subprocess
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from flax import nnx
from lineartgen_runtime import LineartModel
from lineartgen_runtime.images import prepare_input, pyramid_plan, to_bgra
from lineartgen_runtime.sampling import denoise
from PIL import Image, ImageDraw

from lineartgen.evaluation_sampling import inference_with_source, runtime_inputs
from lineartgen.export import export_model
from lineartgen.models.serialization import load_inference, save_inference

BASE = Path("/home/whitegreen/projects/lineartgen/.worktrees")
IDS = [450, 480, 510, 540, 460, 490, 520, 550, 1287, 1383, 717, 727]


def prepare(root):
    assert not root.exists()
    original = (
        BASE
        / "jax-source-signal-20261009/runs/source-signal-20261009/snapshots/step-4000/signal/raw"
    )
    broad = BASE / "jax-source-feature-broad-20261009/runs/source-feature-broad-20261009"
    reference = read(broad / "reference.json")
    manifest = read(Path(reference["root"]) / "cases.json")
    crops = read(broad / "plan.json")["crops"]
    crops.update(
        {
            str(x["case"]): x["crop"]
            for x in read(broad / "native-review/informative-partial-plan.json")["selection"]
        }
    )
    inputs = {}
    for i in IDS:
        p = Path(reference["root"]) / manifest["cases"][i]["file"]
        assert sha(p) == reference["files"][p.name]
        inputs[str(i)] = dict(
            path=str(p),
            sha256=sha(p),
            strength=manifest["cases"][i]["strength"],
            crop=crops[str(i)],
        )
    write(
        root / "plan.json",
        dict(
            original=str(original),
            initial_files={
                n: sha(original / n) for n in ("weights.npz", "config.json", "metadata.json")
            },
            inputs=inputs,
            ids=IDS,
            strengths=[0.1, 0.5],
            steps=[1, 5, 20],
            seeds=[31, 57],
            plan_document_sha256=sha(Path("docs/one-pass-restoration-plan-20261009.md")),
            adoption=False,
        ),
    )


def freeze(root):
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    files = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    subprocess.run(
        ["git", "archive", "--format=tar.gz", "-o", str(root / "source.tar.gz"), commit], check=True
    )
    write(
        root / "source.json",
        dict(
            commit=commit,
            files={
                str(Path(p).resolve()): sha(p)
                for p in files
                if p.endswith((".py", ".toml")) or p == "uv.lock"
            },
            plan_sha256=sha(root / "plan.json"),
            archive_sha256=sha(root / "source.tar.gz"),
        ),
    )


def verify(root):
    source, plan = read(root / "source.json"), read(root / "plan.json")
    assert sha(root / "plan.json") == source["plan_sha256"]
    for p, d in source["files"].items():
        assert sha(p) == d, p
    for p, d in plan["initial_files"].items():
        assert sha(Path(plan["original"]) / p) == d
    for x in plan["inputs"].values():
        assert sha(x["path"]) == x["sha256"]
    return plan


def run(root):
    plan = verify(root)
    original, c = load_inference(plan["original"])
    assert original.feature_input == "source" and original.sampling_mode == "diffusion"
    model = nnx.clone(original)
    model.sampling_mode = "one_pass"
    c = c.model_copy(update={"model": c.model.model_copy(update={"sampling_mode": "one_pass"})})
    jax.tree.map(
        np.testing.assert_array_equal, nnx.state(original, nnx.Param), nnx.state(model, nnx.Param)
    )
    save_inference(
        model,
        c,
        root / "model",
        provenance="Source-signal raw4000; same tensors, explicit one-pass sampling",
    )
    path = export_model(
        model, c, root / "model.onnx", provenance="Private diagnostic one-pass copy; not adopted"
    )
    runtime = LineartModel(path, provider="cpu", num_threads=1)
    assert runtime.sampling_mode == "one_pass"
    forward = nnx.jit(lambda m, x, t, s: inference_with_source(m, x, t, s, steps=1))
    rows, images, arrays = [], {}, {}
    ids = plan["ids"] + ["blank-whole", "blank-partial"]
    for i in ids:
        if isinstance(i, int):
            info = plan["inputs"][str(i)]
            with np.load(info["path"], allow_pickle=False) as d:
                clean = d["image"]
                source = d["source"] if "source" in d else clean
                edit = d["strength"] / info["strength"]
        else:
            clean = source = np.ones((512, 512, 1), np.float32)
            edit = np.ones_like(source)
            if i == "blank-partial":
                edit[:128] = 0
                edit[:, :170] = 0
        gray = np.floor(
            np.clip((source[..., 0] / c.data.image_scale + 1) * 127.5, 0, 255) + 0.5
        ).astype(np.uint8)
        scribble = np.zeros((512, 512, 4), np.uint8)
        scribble[..., 3] = 255 - gray
        lineart = np.repeat(gray[..., None], 4, axis=-1)
        lineart[..., 3] = np.floor(np.clip(1 - edit[..., 0], 0, 1) * 255 + 0.5).astype(np.uint8)
        p = pyramid_plan(512, 512, c.model.minimum_size, 128)
        for strength in plan["strengths"]:
            decoded, mask = prepare_input(
                scribble.tobytes(), lineart.tobytes(), p, strength, c.data.image_scale
            )
            quantization = float(abs(decoded[0] - source).max())
            assert quantization <= c.data.image_scale / 255 + 1e-6
            inputs = jax.device_put(runtime_inputs(decoded, mask, 31, len(p.sizes)))
            reference = np.asarray(forward(original, *inputs, jax.device_put(decoded)))
            predicted = np.asarray(forward(model, *inputs, jax.device_put(decoded)))
            np.testing.assert_array_equal(predicted, reference)
            raw = denoise(
                runtime.forward,
                decoded,
                mask,
                seed=31,
                steps=20,
                levels=len(p.sizes),
                source_conditioning=True,
                preserve_zero_noise=True,
                sampling_mode="one_pass",
            )
            np.testing.assert_allclose(raw, reference, rtol=2e-4, atol=2e-5)
            np.testing.assert_array_equal(raw[mask == 0], decoded[mask == 0])
            expected = np.frombuffer(to_bgra(reference, p), np.uint8).astype(int)
            kwargs = dict(
                scribble=scribble.tobytes(),
                lineart=lineart.tobytes(),
                width=512,
                height=512,
                strength=strength,
                seed=31,
            )
            first = None
            for steps in plan["steps"]:
                output = runtime.infer(**kwargs, denoise_steps=steps)
                if first is None:
                    first = output
                assert output == first
                error = int(np.max(np.abs(np.frombuffer(output, np.uint8).astype(int) - expected)))
                assert error <= 1
                rows.append(
                    dict(
                        case=i,
                        strength=strength,
                        steps=steps,
                        max_float_error=float(abs(raw - reference).max()),
                        max_bgra_error=error,
                        max_source_quantization=quantization,
                        protected_exact=True,
                    )
                )
            assert runtime.infer(**(kwargs | dict(seed=57)), denoise_steps=5) == first
            actual_gray = 255 - np.frombuffer(first, np.uint8).reshape(512, 512, 4)[..., 3]
            target = np.where(mask[0] > 0, clean, decoded[0])

            def to_gray(x):
                return np.rint(np.clip((x[..., 0] + 1) * 127.5, 0, 255)).astype(np.uint8)

            images[str(i), strength] = [
                to_gray(target),
                to_gray(decoded[0]),
                255 - expected.reshape(512, 512, 4)[..., 3].astype(np.uint8),
                actual_gray,
                actual_gray,
            ]
            arrays[f"{i}-{strength}"] = raw[0]
            print(dict(case=i, strength=strength, bgra_error=error), flush=True)
    assert len(rows) == 84
    np.savez_compressed(root / "predictions.npz", **arrays)
    records = []
    groups = {
        "real": IDS[:8],
        "geometry-boundary": IDS[8:],
        "blank": ["blank-whole", "blank-partial"],
    }
    for group, selected in groups.items():
        for strength in plan["strengths"]:
            canvas = Image.new("L", (800, 184 * len(selected)), 255)
            for row, i in enumerate(selected):
                rect = plan["inputs"][str(i)]["crop"] if isinstance(i, int) else [90, 352, 250, 512]
                for col, (label, pixels) in enumerate(
                    zip(
                        ("target", "BGRA source", "JAX one", "ONNX one", "ONNX20"),
                        images[str(i), strength],
                        strict=True,
                    )
                ):
                    canvas.paste(Image.fromarray(pixels).crop(rect), (col * 160, row * 184 + 24))
                    ImageDraw.Draw(canvas).text(
                        (col * 160 + 2, row * 184 + 4), f"{i} {label}", fill=0
                    )
            name = f"{group}-{strength}-native.png"
            canvas.save(root / name)
            records.append(dict(file=name, sha256=sha(root / name)))
    write(
        root / "metrics.json",
        dict(
            rows=rows,
            conditions=84,
            seed_checks=28,
            weights_identical=True,
            parameters=sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param))),
            all_steps_identical=True,
            protected_exact=True,
            source_sha256=sha(root / "source.json"),
            model_sha256=sha(path),
            predictions_sha256=sha(root / "predictions.npz"),
            images=records,
            adoption=False,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "freeze", "run"])
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    globals()[args.action](args.root.resolve())
