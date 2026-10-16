"""Fixed FN/FO weights: distinguish broad-stroke erasure from context transfer."""

import argparse
import subprocess
import time
from pathlib import Path

import jax
import numpy as np
from aligned_recursion_diagnostic import diagnostic_infer, pixels
from campaign_protocol import read, sha, write
from cue_capacity import prepare_inputs
from flax import nnx
from physical_field_model import infer, load_field
from PIL import Image, ImageDraw
from plausible_study import digest
from width_response_study import SEED

PARENT = Path("../jax-wide-source-20261016/runs/wide-source-20261016").resolve()
MODELS = {"FN": PARENT / "reference", "FO": PARENT / "snapshots/step-6000/expanded/raw"}
DOC = Path("docs/wide-recursion-plan-20261016.md")
DEPTHS = tuple(range(1, 7))
ENCODINGS = ("float", "uint8")
MODES = ("normal", "zero")


def prepare(root):
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    assert read(PARENT / "completion.json")["stage"] == "complete_wide_source"
    root.mkdir(parents=True, exist_ok=False)
    records, bindings = [], {}
    for index in (59, 61, 63):
        p = PARENT / f"width-sweep/wide-{index}-factor-2.0.npz"
        with np.load(p) as z:
            y, x = np.where(z["source"][..., 0] < 0.98)
        box = (
            max(0, int(x.min()) - 8),
            max(0, int(y.min()) - 8),
            min(256, int(x.max()) + 9),
            min(256, int(y.max()) + 9),
        )
        records.append(dict(index=index, input=str(p), crop=box))
        bindings[str(p)] = sha(p)
    for folder in MODELS.values():
        bindings.update({str(p): sha(p) for p in folder.iterdir() if p.is_file()})
    for p in (DOC.resolve(), PARENT / "completion.json", PARENT / "diagnostics/decision.json"):
        bindings[str(p)] = sha(p)
    for name in subprocess.check_output(["git", "ls-files"], text=True).splitlines():
        if name.endswith((".py", ".toml")) or name == "uv.lock":
            p = Path(name).resolve()
            bindings[str(p)] = sha(p)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(
        ["git", "archive", "--format=tar.gz", "-o", str(root / "source.tar.gz"), commit], check=True
    )
    write(
        root / "plan.json",
        dict(
            commit=commit,
            records=records,
            models={k: str(v) for k, v in MODELS.items()},
            bindings=bindings,
            source_archive_sha256=sha(root / "source.tar.gz"),
            outputs=144,
            native_pages=12,
            training=False,
            adoption=False,
            fresh_confirmation=False,
        ),
    )


def check(root):
    plan = read(root / "plan.json")
    for p, h in plan["bindings"].items():
        assert sha(Path(p)) == h, p
    assert sha(root / "source.tar.gz") == plan["source_archive_sha256"]
    return plan


def evaluate(root):
    plan = check(root)
    assert jax.devices()[0].platform != "cpu"
    models = {m: load_field(p)[0] for m, p in MODELS.items()}
    states = {m: digest(nnx.state(model, nnx.Param)) for m, model in models.items()}
    out = root / "evaluation"
    out.mkdir(exist_ok=False)
    rows, bindings, pages = [], {}, []
    compared = 0
    started = time.perf_counter()
    for r in plan["records"]:
        with np.load(r["input"]) as z:
            src = z["source"]
            previous = {"FN": z["initial"], "FO": z["expanded"]}
        quant = (
            np.rint(np.clip((src + 1) * 127.5, 0, 255)).astype(np.uint8).astype(np.float32) / 127.5
            - 1
        )
        for encoding, source in (("float", src), ("uint8", quant)):
            arrays = {}
            for depth in DEPTHS:
                prepared = prepare_inputs(
                    source[None],
                    source[None],
                    np.full_like(source[None], 0.5),
                    jax.random.PRNGKey(SEED),
                    depth,
                )
                for m, model in models.items():
                    baseline = np.asarray(infer(model, prepared))[0]
                    if depth == 3 and encoding == "float":
                        np.testing.assert_array_equal(baseline, previous[m])
                        compared += 1
                    for mode in MODES:
                        value, field = diagnostic_infer(model, prepared, zero=mode == "zero")
                        value, field = np.asarray(value)[0], np.asarray(field)[0]
                        assert np.isfinite(value).all() and np.isfinite(field).all()
                        assert value.min() >= -1 and value.max() <= 1
                        arrays[f"{m}_{mode}_{depth}"] = value
                        arrays[f"{m}_{mode}_{depth}_field"] = field
                    delta = float(np.max(abs(arrays[f"{m}_normal_{depth}"] - baseline)))
                    np.testing.assert_allclose(
                        arrays[f"{m}_normal_{depth}"], baseline, atol=5e-6, rtol=1e-4
                    )
                    if depth == 1:
                        np.testing.assert_array_equal(
                            arrays[f"{m}_normal_1"], arrays[f"{m}_zero_1"]
                        )
                    rows.append(
                        dict(
                            index=r["index"],
                            encoding=encoding,
                            model=m,
                            depth=depth,
                            parity_max_abs=delta,
                        )
                    )
            p = out / f"probe{r['index']}-{encoding}.npz"
            np.savez_compressed(p, source=source, **arrays)
            bindings[str(p)] = sha(p)
            box = r["crop"]
            w, h = box[2] - box[0], box[3] - box[1]
            stride = max(w, 130)
            for m in models:
                canvas = Image.new("RGB", (stride * 6, (h + 30) * 3), "white")
                pen = ImageDraw.Draw(canvas)
                for col, depth in enumerate(DEPTHS):
                    for row, (mode, value) in enumerate(
                        [("source", source)]
                        + [(mode, arrays[f"{m}_{mode}_{depth}"]) for mode in MODES]
                    ):
                        x, y = col * stride, row * (h + 30)
                        pen.text((x + 2, y + 4), f"{m} {mode} d{depth}", fill="black")
                        canvas.paste(pixels(value).crop(box), (x, y + 25))
                dest = out / f"native-{r['index']}-{m}-{encoding}.png"
                canvas.save(dest)
                pages.append(
                    dict(
                        file=dest.name,
                        sha256=sha(dest),
                        crop=box,
                        native_scale=True,
                        prediction=str(p),
                        prediction_sha256=sha(p),
                    )
                )
        print(dict(case=r["index"], outputs=48), flush=True)
    assert compared == 6
    assert states == {m: digest(nnx.state(model, nnx.Param)) for m, model in models.items()}
    write(
        out / "metrics.json",
        dict(
            rows=rows,
            outputs=144,
            existing_predictions_exact=compared,
            states_unchanged=True,
            normal_parity_passed=True,
            depth1_modes_exact=True,
            seconds=time.perf_counter() - started,
            bindings=bindings,
        ),
    )
    write(
        out / "manifest.json",
        dict(records=pages, native_scale=True, thresholding=False, contrast_rescaling=False),
    )


def finalize(root):
    check(root)
    out = root / "evaluation"
    metrics = read(out / "metrics.json")
    assert metrics["outputs"] == 144 and metrics["existing_predictions_exact"] == 6
    assert (
        metrics["states_unchanged"]
        and metrics["normal_parity_passed"]
        and metrics["depth1_modes_exact"]
    )
    for p, h in metrics["bindings"].items():
        assert sha(Path(p)) == h
    review = read(out / "visual-review.json")
    records = read(out / "manifest.json")["records"]
    assert review["reviewed"] and len(records) == len(review["observations"]) == 12
    for r in records:
        assert sha(out / r["file"]) == r["sha256"] == review["files"][r["file"]]
        assert sha(Path(r["prediction"])) == r["prediction_sha256"]
        assert r["file"] in review["observations"]
    assert read(root / "driver-evaluate.json")["returncode"] == 0
    files = [
        "plan.json",
        "evaluation/metrics.json",
        "evaluation/manifest.json",
        "evaluation/visual-review.json",
        "decision.json",
        "driver-evaluate.json",
    ]
    write(
        root / "completion.json",
        dict(
            stage="complete_wide_recursion",
            outputs=144,
            native_pages_reviewed=12,
            training=False,
            adoption=False,
            fresh_confirmation=False,
            evidence={f: sha(root / f) for f in files},
        ),
    )
    print("Completed wide recursion diagnostic", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "evaluate", "finalize"))
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    globals()[args.action](args.root.resolve())
