"""Render whole-image geometry cases selected before observing candidate outputs."""

import argparse
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from evaluate_lineart_loss_study import render
from flax import nnx
from train_lineart_loss_study import verify

from lineartgen.diffusion import sample_prepared
from lineartgen.evaluation import probe_metrics, protected_metrics
from lineartgen.evaluation_sampling import runtime_inputs
from lineartgen.lineart_metrics import halo_metrics
from lineartgen.models.serialization import load_inference
from lineartgen.training.lineart_study import METHODS


def evaluate(root, step, mode):
    protocol = verify(root)
    selection_path = root / "supplemental-panels.json"
    selected = read(selection_path)
    assert selected["selected_before_candidate_evaluation"]
    reference = read(root / "evaluation.json")
    cases = Path(reference["root"])
    manifest = read(cases / "cases.json")
    assert sha(cases / "cases.json") == selected["manifest_sha256"] == reference["cases_sha256"]
    paths = {"initial": Path(protocol["initial"])}
    if step:
        paths.update({name: root / "snapshots" / f"step-{step}" / name / mode for name in METHODS})
    hashes = {name: sha(path / "weights.npz") for name, path in paths.items()}
    folder = root / "supplemental-evaluation" / f"step-{step}-{mode}"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "metrics.json").exists():
        previous = read(folder / "metrics.json")
        assert previous["model_weights_sha256"] == hashes
        assert previous["selection_sha256"] == sha(selection_path)
        assert previous["analysis_source_sha256"] == sha(Path(__file__))
        return
    models = {name: load_inference(path)[0] for name, path in paths.items()}
    inference = nnx.jit(lambda m, x, t: sample_prepared(m, x, t, steps=1))
    rows = []
    for requested in selected["cases"]:
        index = requested["case"]
        case = manifest["cases"][index]
        assert not case["partial"]
        assert sha(cases / case["file"]) == reference["files"][case["file"]]
        for key in ("kind", "family", "severity", "strength", "variant"):
            assert case[key] == requested[key]
        with np.load(cases / case["file"], allow_pickle=False) as arrays:
            target, source, strength = arrays["image"], arrays["source"], arrays["strength"]
            probes, connections = arrays["probes"], arrays["connections"]
        prepared = runtime_inputs(source[None], strength[None], case["seed"], levels=3)
        noisy = prepared[0][0][0]
        prepared = jax.device_put(prepared)
        predictions = {}
        for name, model in models.items():
            prediction = np.asarray(inference(model, *prepared))[0]
            assert np.isfinite(prediction).all()
            measured = (
                protected_metrics(prediction, target, strength)
                | halo_metrics(prediction, target, strength)
                | probe_metrics(prediction, target, probes, connections)
            )
            assert all(value is None or np.isfinite(value) for value in measured.values())
            rows.append(dict(**requested, method=name, **measured))
            predictions[name] = prediction
        render(folder, index, 1, target, source, noisy, predictions)
    write(
        folder / "metrics.json",
        dict(
            step=step,
            mode=mode,
            prediction_count=len(rows),
            results=rows,
            model_weights_sha256=hashes,
            selection_sha256=sha(selection_path),
            analysis_source_sha256=sha(Path(__file__)),
            denoise_steps=1,
            recursion_levels=3,
            limitation="Fixed supplemental visual panels; reused synthetic development cases",
        ),
    )
    print(f"Rendered {len(selected['cases'])} whole-geometry panels")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--mode", choices=("ema", "raw"), default="ema")
    args = parser.parse_args()
    evaluate(args.root.resolve(), args.step, args.mode)


if __name__ == "__main__":
    main()
