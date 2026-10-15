"""Native 512px screen using the frozen established scenes; not fresh confirmation."""

import argparse
from pathlib import Path

import numpy as np
from campaign_protocol import read, sha, write
from quartic_sign_study import REPORT_METHODS as METHODS
from quartic_sign_study import STEPS, check

SIZE = 512
SEED = 2026101319


def evaluate():
    import jax
    from cue_capacity import prepare_inputs
    from physical_field_model import infer, load_field

    check(ROOT)
    plan = read(OUT / "plan.json")
    paths = {m: ROOT / f"snapshots/step-{STEPS}/{m}/raw" for m in METHODS} | dict(
        initial=ROOT / "reference"
    )
    models = {m: load_field(p)[0] for m, p in paths.items()}
    bindings = {
        str(p): sha(p) for folder in paths.values() for p in folder.iterdir() if p.is_file()
    }
    yy, xx = np.indices((SIZE, SIZE))
    rows = []
    for record in plan["records"]:
        i = record["index"]
        assert sha(OUT / record["file"]) == record["sha256"]
        with np.load(OUT / record["file"]) as z:
            source, quantized, clean = z["source"], z["quantized"], z["clean"]
        for mode in plan["modes"]:
            t = (
                np.full_like(source, 0.5)
                if mode == "whole"
                else np.where((xx > SIZE * 0.6 - yy * 0.25)[..., None], 0.5, 0).astype(np.float32)
            )
            for draw in range(2):
                key = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(SEED), i), draw)
                predictions = {}
                for encoding, input_ in (("float", source), ("uint8", quantized)):
                    p = prepare_inputs(clean[None], input_[None], t[None], key, depth=6)
                    for method, model in models.items():
                        pred = np.asarray(infer(model, p))[0]
                        assert np.isfinite(pred).all() and pred.min() >= -1 and pred.max() <= 1
                        np.testing.assert_array_equal(pred[t == 0], input_[t == 0])
                        predictions[f"{method}_{encoding}"] = pred
                file = OUT / f"predictions-{i}-{mode}-{draw}.npz"
                np.savez_compressed(file, **predictions)
                bindings[str(file)] = sha(file)
                rows.append(
                    dict(
                        index=i,
                        condition=record["condition"],
                        mode=mode,
                        draw=draw,
                        quantization_output_rms={
                            m: float(
                                np.sqrt(
                                    np.mean(
                                        (predictions[m + "_uint8"] - predictions[m + "_float"]) ** 2
                                    )
                                )
                            )
                            for m in models
                        },
                    )
                )
        print(dict(scene=i + 1, stage="scene512"), flush=True)
    write(
        OUT / "metrics.json",
        dict(rows=rows, model_outputs=192, protected_exact=True, bindings=bindings),
    )
    render()


def render():
    from PIL import Image, ImageDraw

    tasks = [("whole", (i, i + 1)) for i in range(0, 8, 2)] + [
        ("partial", (1, 3)),
        ("partial", (5, 7)),
    ]
    records = []
    for mode, indices in tasks:
        canvas = Image.new("RGB", (5 * SIZE, 36 + 2 * (SIZE + 24)), "white")
        pen = ImageDraw.Draw(canvas)
        for col, label in enumerate(
            (
                "uint8 rough source",
                "EQ mixed raw18000 uint8",
                "EX physical +6000 uint8",
                "quartic +6000 uint8",
                "quartic +6000 float",
            )
        ):
            pen.text((col * SIZE + 8, 8), label, fill="black")
        for row, i in enumerate(indices):
            with (
                np.load(OUT / f"input-{i}.npz") as inp,
                np.load(OUT / f"predictions-{i}-{mode}-0.npz") as pred,
            ):
                for col, value in enumerate(
                    (
                        inp["quantized"],
                        pred["initial_uint8"],
                        pred["control_uint8"],
                        pred["quartic_uint8"],
                        pred["quartic_float"],
                    )
                ):
                    gray = np.rint(np.clip((value[..., 0] + 1) / 2, 0, 1) * 255).astype(np.uint8)
                    canvas.paste(
                        Image.fromarray(gray).convert("RGB"), (col * SIZE, 60 + row * (SIZE + 24))
                    )
            pen.text((8, 38 + row * (SIZE + 24)), f"scene {i} / {mode} / depth6", fill="black")
        file = OUT / f"native-{mode}-{indices[0]}-{indices[1]}.png"
        canvas.save(file)
        records.append(dict(file=file.name, sha256=sha(file)))
    write(
        OUT / "manifest.json",
        dict(records=records, native_size=512, thresholding=False, contrast_rescaling=False),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    ROOT = a.root.resolve()
    OUT = ROOT / "scene512"
    evaluate()
