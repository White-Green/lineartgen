"""Local I3k integration fixture; never installs or selects a distribution model."""

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


def run(research, root):
    from component_readout_runtime import forward as reference_forward

    assert not root.exists()
    train = research / "runs/component-joint-fit-20261112/export-3000/joint/raw"
    evaluation = research / "runs/component-joint-results-3000-20261112/joint-raw"
    verification = json.loads((evaluation / "onnx/verification.json").read_text())
    assert verification["complete"]
    assert verification["onnx_sha256"] == sha(evaluation / "onnx/stage.onnx")
    onnx_plan = json.loads((evaluation / "onnx/plan.json").read_text())
    assert onnx_plan["bindings"][str((train / "weights.npz").resolve())] == sha(
        train / "weights.npz"
    )
    runtime = (
        Path(__file__).resolve().parents[1] / "packages/lineartgen-runtime/src/lineartgen_runtime"
    )
    files = [
        Path(__file__),
        train / "weights.npz",
        train / "readout.npz",
        evaluation / "onnx/stage.onnx",
        evaluation / "onnx/verification.json",
        evaluation / "expanded/results.json",
        research / "scripts/component_readout_runtime.py",
        research / "scripts/cnn_component_projection.py",
    ]
    files += list(runtime.glob("*.py")) + [runtime / "assets/spatial-stage-model.json"]
    consumed = json.loads((evaluation / "expanded/results.json").read_text())
    assert consumed["complete"]
    indexed = {(r["case"], r["tone"], r["painted"]): r for r in consumed["records"]}
    cases = [
        ("precision-s1-021-c1-z0.5", 0, "main"),
        ("precision-s1-025-c1-z2", 1, "none"),
        ("precision-s1-002-c2-z2", 2, "white"),
    ]
    for case in cases:
        row = indexed[case]
        path = evaluation / "expanded" / row["file"]
        assert sha(path) == row["sha256"]
        files.append(path)
    bindings = {str(p.resolve()): sha(p) for p in files}
    write(
        root / "plan.json",
        dict(
            bindings=bindings,
            cases=cases,
            local_fixture_only=True,
            adopted=False,
            no_confirmation=True,
            no_push=True,
            final_candidate_checks_still_required=True,
        ),
    )
    bundle = root / "fixture"
    bundle.mkdir()
    shutil.copyfile(evaluation / "onnx/stage.onnx", bundle / "spatial-stage.onnx")
    shutil.copyfile(train / "readout.npz", bundle / "component-readout.npz")
    spec = json.loads((runtime / "assets/spatial-stage-model.json").read_text())
    spec = {k: v for k, v in spec.items() if not k.endswith("_training_steps")}
    spec.update(
        format_version=2,
        component_projection="base_component_mean_max_mlp32",
        parameters=140275,
        cnn_parameters=139186,
        readout_parameters=1089,
        checkpoint_sha256=sha(train / "weights.npz"),
        local_verification_fixture_only=True,
        fixture_source_updates=3000,
        candidate_adopted=False,
        stage=dict(file="spatial-stage.onnx", sha256=sha(bundle / "spatial-stage.onnx")),
        readout=dict(file="component-readout.npz", sha256=sha(bundle / "component-readout.npz")),
    )
    manifest = bundle / "spatial-stage-model.json"
    write(manifest, spec)
    model = StageLineartModel(manifest, num_threads=1, provider="cpu")
    public = LineartModel(manifest, num_threads=1, provider="cpu")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    reference_session = TiledSelectionSession(
        ort.InferenceSession(
            str(bundle / "spatial-stage.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    )
    with np.load(train / "readout.npz", allow_pickle=False) as z:
        params = dict(z)
    records = []
    for case_index, case in enumerate(cases):
        row = indexed[case]
        with np.load(evaluation / "expanded" / row["file"], allow_pickle=False) as z:
            values = dict(z)
        scribble, lineart = values["scribble"], values["lineart"]
        h, w, _ = scribble.shape
        seed = int(
            np.random.default_rng(
                [2026110907, 1, row["group"], int(row["zoom"] * 2), row["tone"]]
            ).integers(2**32)
        )
        args = dict(
            scribble=scribble.tobytes(),
            lineart=lineart.tobytes(),
            width=w,
            height=h,
            strength=0.5,
            seed=seed,
            denoise_steps=6,
        )
        output = model.infer(**args)
        actual = np.frombuffer(output, np.uint8).reshape(h, w, 4)
        np.testing.assert_array_equal(actual[..., 3].astype(np.float32) / 255, values["alpha"])
        assert not actual[..., :3].any()
        opaque = lineart[..., 3] == 255
        np.testing.assert_array_equal(actual[..., 3][opaque], 255 - lineart[..., 0][opaque])
        depths = []
        if case_index == 0:
            assert output == model.infer(**args) == public.infer(**args)
            for depth in range(1, 7):
                levels, _, _ = levels_from_bgra(
                    args["scribble"], args["lineart"], w, h, 0.5, seed, depth
                )
                expected = reference_forward(reference_session, levels, params=params)
                result = forward(model._get_session(), levels, readout=model._readout)
                for before, after in zip(expected, result, strict=True):
                    np.testing.assert_array_equal(before, after)
                assert model.infer(**(args | dict(denoise_steps=depth))) == public.infer(
                    **(args | dict(denoise_steps=depth))
                )
                depths.append(depth)
        records.append(
            dict(
                case=case,
                shape=[h, w],
                saved_development_alpha_exact=True,
                opaque_pixels_exact=int(opaque.sum()),
                recursive_ONNX_reference_exact=depths,
                output_sha256=hashlib.sha256(output).hexdigest(),
            )
        )
        print("Readout runtime fixture exact", case, (h, w), flush=True)
    # Exercise the public BGRA fast paths and reject corrupt/mismatched assets.
    h, w = 65, 97
    empty = np.zeros((h, w, 4), np.uint8)
    args = dict(
        scribble=empty.tobytes(),
        lineart=empty.tobytes(),
        width=w,
        height=h,
        strength=0.5,
        seed=2**40 + 123,
        denoise_steps=6,
    )
    assert public.infer(**args) == empty.tobytes()
    rough = empty.copy()
    rough[..., 3] = np.arange(w, dtype=np.uint8)
    assert public.infer(**(args | dict(scribble=rough.tobytes(), strength=0))) == rough.tobytes()
    painted = empty.copy()
    painted[:30] = [0, 0, 0, 255]
    painted[30:] = [255, 255, 255, 255]
    result = np.frombuffer(public.infer(**(args | dict(lineart=painted.tobytes()))), np.uint8)
    result = result.reshape(h, w, 4)
    np.testing.assert_array_equal(result[..., 3], 255 - painted[..., 0])
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
    assert "jax" not in sys.modules and "flax" not in sys.modules
    assert all(sha(p) == h for p, h in bindings.items())
    write(
        root / "verification.json",
        dict(
            complete=True,
            bindings=bindings,
            records=records,
            blank_exact=True,
            opaque_black_white_exact=True,
            strength_zero_exact=True,
            fixed_seed_exact=True,
            public_API_exact=True,
            invalid_manifests_rejected=rejections,
            no_JAX_Flax_import=True,
            no_confirmation=True,
            quality_adopted=False,
            final_candidate_checks_still_required=True,
            actual_windows_directml_tested=False,
            actual_krita_tested=False,
            no_distributable_artifact=True,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run(args.research.resolve(), args.root.resolve())
