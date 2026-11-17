"""Check the frozen S12 candidate in the prepared runtime without changing bundled assets."""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

from lineartgen_runtime import LineartModel
from lineartgen_runtime.curve_inputs import levels_from_bgra
from lineartgen_runtime.selection_feedback import TiledSelectionSession, forward
from lineartgen_runtime.spatial_stage import StageLineartModel


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(research, candidate_path, root):
    from component_readout_runtime import forward as reference_forward

    assert not root.exists()
    candidate = json.loads(candidate_path.read_text())
    assert candidate["selected_updates"] == 12000 and candidate["mode"] == "ema"
    assert all(sha(p) == h for p, h in candidate["bindings"].items())
    train = Path(candidate["weights"]).parent
    verified, reference = Path(candidate["verified"]), Path(candidate["reference"])
    assert sha(verified / "stage.onnx") == candidate["onnx_sha256"]
    assert sha(train / "readout.npz") == candidate["readout_sha256"]
    runtime = Path(__file__).resolve().parents[1] / "packages/lineartgen-runtime/src/lineartgen_runtime"
    results = json.loads((reference / "results.json").read_text())
    assert results["complete"]
    indexed = {(r["case"], r["tone"], r["painted"]): r for r in results["records"]}
    cases = []
    for group in range(9):
        choice = group % 3
        zoom = ("0.5", "1", "2")[group % 3]
        painted = "none" if group % 4 == 0 else ("main", "soft", "white")[group % 3]
        cases.append((f"precision-s1-{group:03d}-c{choice}-z{zoom}", group % 3, painted))
    files = [Path(__file__), candidate_path, train / "weights.npz", train / "readout.npz",
             verified / "stage.onnx", verified / "verification.json", reference / "results.json",
             research / "scripts/component_readout_runtime.py", research / "scripts/cnn_component_projection.py"]
    files += list(runtime.glob("*.py")) + [runtime / "assets/spatial-stage-model.json"]
    for case in cases:
        row = indexed[case]
        path = reference / row["file"]
        assert sha(path) == row["sha256"]
        files.append(path)
    bindings = {str(p.resolve()): sha(p) for p in files}
    write(root / "plan.json", dict(bindings=bindings, cases=cases, selected_updates=12000,
          local_fixture_only=True, no_confirmation=True, no_distributable_artifact=True))
    bundle = root / "fixture"
    bundle.mkdir()
    shutil.copyfile(verified / "stage.onnx", bundle / "spatial-stage.onnx")
    shutil.copyfile(train / "readout.npz", bundle / "component-readout.npz")
    spec = json.loads((runtime / "assets/spatial-stage-model.json").read_text())
    spec = {k: v for k, v in spec.items() if not k.endswith("_training_steps")}
    spec.update(format_version=2, component_projection="base_component_mean_max_mlp32",
        parameters=140275, cnn_parameters=139186, readout_parameters=1089,
        checkpoint_sha256=sha(train / "weights.npz"), candidate_adopted=False,
        selected_training_updates=12000, training_evaluated_until=20000,
        stage=dict(file="spatial-stage.onnx", sha256=sha(bundle / "spatial-stage.onnx")),
        readout=dict(file="component-readout.npz", sha256=sha(bundle / "component-readout.npz")))
    manifest = bundle / "spatial-stage-model.json"
    write(manifest, spec)
    model = StageLineartModel(manifest, num_threads=1, provider="cpu")
    public = LineartModel(manifest, num_threads=1, provider="cpu")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    reference_session = TiledSelectionSession(ort.InferenceSession(
        str(bundle / "spatial-stage.onnx"), sess_options=options, providers=["CPUExecutionProvider"]))
    with np.load(train / "readout.npz", allow_pickle=False) as z:
        params = dict(z)
    records = []
    for index, case in enumerate(cases):
        row = indexed[case]
        with np.load(reference / row["file"], allow_pickle=False) as z:
            values = dict(z)
        scribble, lineart = values["scribble"], values["lineart"]
        h, w, _ = scribble.shape
        seed = int(np.random.default_rng([2026110907, 1, row["group"],
            int(row["zoom"] * 2), row["tone"]]).integers(2**32))
        args = dict(scribble=scribble.tobytes(), lineart=lineart.tobytes(), width=w, height=h,
                    strength=.5, seed=seed, denoise_steps=6)
        output = model.infer(**args)
        actual = np.frombuffer(output, np.uint8).reshape(h, w, 4)
        np.testing.assert_array_equal(actual[..., 3].astype(np.float32) / 255, values["alpha"])
        assert not actual[..., :3].any()
        opaque = lineart[..., 3] == 255
        np.testing.assert_array_equal(actual[..., 3][opaque], 255 - lineart[..., 0][opaque])
        depths = []
        if index == 0:
            assert output == model.infer(**args) == public.infer(**args)
            for depth in range(1, 7):
                levels, _, _ = levels_from_bgra(args["scribble"], args["lineart"], w, h,
                                               .5, seed, depth)
                expected = reference_forward(reference_session, levels, params=params)
                result = forward(model._get_session(), levels, readout=model._readout)
                for before, after in zip(expected, result, strict=True):
                    np.testing.assert_array_equal(before, after)
                assert model.infer(**(args | dict(denoise_steps=depth))) == public.infer(
                    **(args | dict(denoise_steps=depth)))
                depths.append(depth)
        records.append(dict(case=case, shape=[h, w], saved_development_alpha_exact=True,
            opaque_pixels_exact=int(opaque.sum()), recursive_ONNX_reference_exact=depths,
            output_sha256=hashlib.sha256(output).hexdigest()))
        print("Frozen candidate runtime exact", case, (h, w), flush=True)
    h, w = 65, 97
    empty = np.zeros((h, w, 4), np.uint8)
    args = dict(scribble=empty.tobytes(), lineart=empty.tobytes(), width=w, height=h,
                strength=.5, seed=2**40 + 123, denoise_steps=6)
    assert public.infer(**args) == empty.tobytes()
    rough = empty.copy()
    rough[..., 3] = np.arange(w, dtype=np.uint8)
    assert public.infer(**(args | dict(scribble=rough.tobytes(), strength=0))) == rough.tobytes()
    painted = empty.copy()
    painted[:30] = [0, 0, 0, 255]
    painted[30:] = [255, 255, 255, 255]
    actual = np.frombuffer(public.infer(**(args | dict(lineart=painted.tobytes()))), np.uint8)
    np.testing.assert_array_equal(actual.reshape(h, w, 4)[..., 3], 255 - painted[..., 0])
    rejections = []
    for name, change in [
        ("readout_hash", dict(readout=dict(spec["readout"], sha256="0" * 64))),
        ("parameter_count", dict(readout_parameters=33)),
        ("version", dict(format_version=1)),
        ("readout_path", dict(readout=dict(spec["readout"], file="../readout.npz"))),
    ]:
        path = bundle / f"invalid-{name}.json"
        write(path, spec | change)
        try:
            StageLineartModel(path, num_threads=1, provider="cpu")
        except ValueError:
            rejections.append(name)
        else:
            raise AssertionError(f"Accepted invalid manifest: {name}")
    assert not any(k == "jax" or k.startswith("jax.") or k == "flax" or k.startswith("flax.")
                   for k in sys.modules)
    assert all(sha(p) == h for p, h in bindings.items())
    write(root / "verification.json", dict(complete=True, bindings=bindings, records=records,
        blank_exact=True, opaque_black_white_exact=True, strength_zero_exact=True, fixed_seed_exact=True,
        public_API_exact=True, invalid_manifests_rejected=rejections, no_JAX_Flax_import=True,
        no_confirmation=True, quality_adopted=False, actual_windows_directml_tested=False,
        actual_krita_tested=False, no_distributable_artifact=True))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--research", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    run(a.research.resolve(), a.candidate.resolve(), a.root.resolve())
