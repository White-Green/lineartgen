"""Synthetic BGRA quality diagnostic for fixed, already rejected development models.

This does not select a model or change the active guide-encoding experiment.
All quality measurements here use the final thresholded BGRA output.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from campaign_protocol import read, sha, write
from lineartgen_runtime import LineartModel
from lineartgen_runtime.images import prepare_input, pyramid_plan
from PIL import Image, ImageDraw

from lineartgen.evaluation import probe_metrics, protected_metrics
from lineartgen.evaluation_sampling import background_metrics
from lineartgen.export import export_model
from lineartgen.lineart_metrics import halo_metrics
from lineartgen.models.serialization import load_inference


def diagnose(root, reference):
    assert not root.exists()
    root.mkdir(parents=True)
    control = Path(read(reference / "plan.json")["control"])
    paths = {
        "control": control / "snapshots/step-2000/control/ema",
        "candidate": reference / "snapshots/step-2000/corruption_blend0625/ema",
    }
    evaluation = read(reference / "evaluation.json")
    cases = Path(evaluation["root"])
    assert sha(cases / "cases.json") == evaluation["cases_sha256"]
    manifest = read(cases / "cases.json")
    selected = read(reference / "supplemental-panels.json")
    assert selected["selected_before_candidate_evaluation"]
    assert selected["manifest_sha256"] == evaluation["cases_sha256"]
    assert len(selected["cases"]) == 24
    models = {
        name: dict(
            path=str(p),
            files={n: sha(p / n) for n in ("weights.npz", "metadata.json", "config.json")},
        )
        for name, p in paths.items()
    }
    write(
        root / "plan.json",
        dict(
            models=models,
            cases=selected["cases"],
            evaluation=evaluation,
            input_scales=[1.1, 1.0],
            denoise_steps=[1, 2, 5],
            script_sha256=sha(Path(__file__)),
            input=(
                "Render each existing rough array as black BGRA ink with "
                "alpha=round((1-source)*127.5); empty lineart layer, whole-image edit. "
                "These are explicit synthetic Krita inputs."
            ),
            scope=(
                "Preprocessing diagnostic on 24 already reused geometry cases; "
                "no acceptance gates, "
                "no training or current-plan changes. Candidate was rejected at independent seed. "
                "Do not compare these thresholded BGRA metrics directly with array-space metrics."
            ),
        ),
    )
    runtimes, exports = {}, {}
    for name, path in paths.items():
        model, config = load_inference(path)
        assert config.data.image_scale == 1.1
        exported = export_model(model, config, root / f"{name}-scale11.onnx", provenance=str(path))
        original = onnx.load(exported)
        graph = original.graph.SerializeToString()
        runtimes[f"{name}_s11"] = LineartModel(exported, provider="cpu", num_threads=1)
        for item in original.metadata_props:
            if item.key == "lineartgen":
                metadata = json.loads(item.value)
                assert metadata["image_scale"] == 1.1
                metadata["image_scale"] = 1.0
                item.value = json.dumps(metadata)
        assert original.graph.SerializeToString() == graph
        linear = root / f"{name}-scale10.onnx"
        onnx.save_model(original, linear)
        runtimes[f"{name}_s10"] = LineartModel(linear, provider="cpu", num_threads=1)
        exports[name] = dict(
            original_sha256=sha(exported),
            linear_sha256=sha(linear),
            only_input_scale_metadata_changed=True,
        )
    rows, inputs = [], []
    canvas_plan = pyramid_plan(512, 512)
    for offset, declaration in enumerate(selected["cases"]):
        index = declaration["case"]
        case = manifest["cases"][index]
        assert not case["partial"] and case["strength"] in (0.1, 0.5)
        file = cases / case["file"]
        assert sha(file) == evaluation["files"][case["file"]]
        with np.load(file, allow_pickle=False) as arrays:
            target, source = arrays["image"], arrays["source"]
            probes, connections = arrays["probes"], arrays["connections"]
        scribble = np.zeros((512, 512, 4), np.uint8)
        scribble[..., 3] = np.rint(np.clip((1 - source[..., 0]) * 127.5, 0, 255)).astype(np.uint8)
        scribble_bytes, lineart_bytes = scribble.tobytes(), bytes(scribble.nbytes)
        input_png = root / f"case-{index:04d}-scribble.png"
        Image.fromarray(scribble, "RGBA").save(input_png)  # RGB is black, so BGRA/RGBA coincide.
        prepared = {}
        for scale in (1.1, 1.0):
            x, level = prepare_input(
                scribble_bytes, lineart_bytes, canvas_plan, case["strength"], scale
            )
            prepared[scale] = x[0]
            inputs.append(
                dict(
                    case=index,
                    scale=scale,
                    nonzero_input_pixels=int(np.count_nonzero(scribble[..., 3])),
                    surviving_pixels=int(np.count_nonzero(x < 1)),
                    completely_erased=bool(not np.any(x < 1)),
                    input_png_sha256=sha(input_png),
                )
            )
        for steps in (1, 2, 5):
            predictions = {}
            for name, runtime in runtimes.items():
                result = runtime.infer(
                    scribble=scribble_bytes,
                    lineart=lineart_bytes,
                    width=512,
                    height=512,
                    strength=case["strength"],
                    seed=case["seed"],
                    denoise_steps=steps,
                )
                bgra = np.frombuffer(result, np.uint8).reshape(512, 512, 4)
                assert not bgra[..., :3].any()
                prediction = 1 - bgra[..., 3:4].astype(np.float32) * (2 / 255)
                level = np.full_like(target, case["strength"])
                metrics = (
                    protected_metrics(prediction, target, level)
                    | background_metrics(prediction, target, level)
                    | halo_metrics(prediction, target, level)
                    | probe_metrics(prediction, target, probes, connections)
                )
                assert all(v is None or np.isfinite(v) for v in metrics.values())
                rows.append(
                    dict(
                        case=index,
                        method=name,
                        kind=case["kind"],
                        family=case["family"],
                        severity=case["severity"],
                        strength=case["strength"],
                        denoise_steps=steps,
                        **metrics,
                    )
                )
                predictions[name] = prediction
            images = {
                "target": target,
                "scribble": 1 - scribble[..., 3:4].astype(np.float32) * (2 / 255),
                "prepared1.1": prepared[1.1],
                "prepared1.0": prepared[1.0],
                **predictions,
            }
            # Native pixels around the fixed teacher probes; no stretching or contrast adjustment.
            x, y = np.median(probes[:, :2], axis=0).round().astype(int)
            x, y = int(np.clip(x - 64, 0, 384)), int(np.clip(y - 48, 0, 416))
            panel = Image.new("L", (128 * len(images), 120), 255)
            for column, (label, value) in enumerate(images.items()):
                pixels = np.rint(np.clip((value[..., 0] + 1) * 127.5, 0, 255)).astype(np.uint8)
                panel.paste(Image.fromarray(pixels[y : y + 96, x : x + 128]), (column * 128, 24))
                ImageDraw.Draw(panel).text((column * 128 + 2, 3), label, fill=0)
            panel.save(root / f"case-{index:04d}-steps-{steps}-native.png")
        print(dict(cases=offset + 1, total=24), flush=True)
    for name, path in paths.items():
        assert models[name]["files"] == {n: sha(path / n) for n in models[name]["files"]}
    write(
        root / "results.json",
        dict(
            plan_sha256=sha(root / "plan.json"),
            exports=exports,
            inputs=inputs,
            prediction_count=len(rows),
            results=rows,
            thresholds_applied_by_actual_bgra_api=True,
            model_selection_or_adoption=False,
        ),
    )
    print(
        dict(
            predictions=len(rows),
            erased_by_scale={
                str(s): sum(r["completely_erased"] for r in inputs if r["scale"] == s)
                for s in (1.1, 1.0)
            },
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.root.resolve(), args.reference.resolve())
