"""Fixed-weight comparison of coarse convolution dilation schedules."""

import argparse
import subprocess
import time
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from cue_capacity import prepare_inputs
from flax import nnx
from PIL import Image, ImageDraw
from plausible_study import digest
from scale_dilation_model import MODES, infer_dilation
from subpixel_field_model import infer, load_field

PARENT = Path("../jax-joint-geometry-20261015/runs/joint-geometry-20261015").resolve()
DOC = Path("docs/scale-dilation-plan-20261015.md")
ET = Path("../jax-context-dropout-20261015/runs/context-dropout-20261015").resolve()
PATHS = dict(
    mixed=PARENT / "snapshots/step-18000/mixed/raw", dropout=ET / "snapshots/step-6000/dropout/raw"
)
ER = Path("../jax-recursion-diagnostic-20261015/runs/recursion-diagnostic-20261015").resolve()
ENCODINGS = ("float", "uint8")


def prepare(root):
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    assert read(PARENT / "completion.json")["stage"] == "complete_joint_geometry"
    assert read(ET / "completion.json")["stage"] == "complete_context_dropout"
    root.mkdir(parents=True, exist_ok=False)
    assert read(ER / "completion.json")["stage"] == "complete_recursion_diagnostic"
    records = read(ER / "plan.json")["records"]
    bound = {}
    for r in records:
        bound[r["input"]] = sha(Path(r["input"]))
        for depth in (1, 6):
            f = ER / f"evaluation/{r['name']}-depth{depth}.npz"
            bound[str(f)] = sha(f)
    for folder in PATHS.values():
        bound.update({str(p): sha(p) for p in folder.iterdir() if p.is_file()})
    for p in (
        DOC.resolve(),
        PARENT / "completion.json",
        PARENT / "diagnostics/decision.json",
        ET / "completion.json",
    ):
        bound[str(p)] = sha(p)
    for name in subprocess.check_output(["git", "ls-files"], text=True).splitlines():
        if name.endswith((".py", ".toml")) or name == "uv.lock":
            p = Path(name).resolve()
            bound[str(p)] = sha(p)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(
        ["git", "archive", "--format=tar.gz", "-o", str(root / "source.tar.gz"), commit], check=True
    )
    write(
        root / "plan.json",
        dict(
            commit=commit,
            records=records,
            modes=MODES,
            depth=6,
            models={m: str(p) for m, p in PATHS.items()},
            encodings=ENCODINGS,
            bindings=bound,
            source_archive_sha256=sha(root / "source.tar.gz"),
            outputs=72,
            native_pages=16,
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


def data(record):
    with np.load(record["input"]) as z:
        source = z["source"]
        clean = z["variants"][13] if record["size"] == 256 else z["clean"]
        quantized = (
            np.rint(np.clip((source + 1) * 127.5, 0, 255)).astype(np.uint8).astype(np.float32)
            / 127.5
            - 1
        )
        if record["size"] == 512:
            np.testing.assert_array_equal(quantized, z["quantized"])
    if record["size"] == 256:
        key = jax.random.PRNGKey(2026101503)
        for value in (3, record["index"], 1, 0):
            key = jax.random.fold_in(key, value)
    else:
        key = jax.random.fold_in(
            jax.random.fold_in(jax.random.PRNGKey(2026101319), record["index"]), 0
        )
    return source, quantized, clean, key


def evaluate(root):
    plan = check(root)
    assert jax.devices()[0].platform != "cpu"
    models = {m: load_field(p)[0] for m, p in PATHS.items()}
    states = {m: digest(nnx.state(model, nnx.Param)) for m, model in models.items()}
    out = root / "evaluation"
    out.mkdir(exist_ok=False)
    rows, bindings, compared = [], {}, 0
    depth1_matches = 0
    prior_matches = 0
    started = time.perf_counter()
    for r in plan["records"]:
        source, quantized, clean, key = data(r)
        t = np.full_like(source[None], 0.5)
        previous = {}
        for mode in MODES:
            arrays = {}
            for encoding, src in (("float", source), ("uint8", quantized)):
                p = prepare_inputs(clean[None], src[None], t, key, 6)
                for m, model in models.items():
                    value = np.asarray(infer_dilation(model, p, mode=mode))[0]
                    assert value.shape == source.shape and np.isfinite(value).all()
                    assert value.min() >= -1 and value.max() <= 1
                    name = f"{m}_{encoding}"
                    arrays[name] = value
                    if mode == "legacy":
                        np.testing.assert_array_equal(value, np.asarray(infer(model, p))[0])
                        compared += 1
                        if m == "mixed":
                            with np.load(ER / f"evaluation/{r['name']}-depth6.npz") as old:
                                np.testing.assert_array_equal(value, old[name])
                            prior_matches += 1
                    if r["name"] in ("probe37", "scene1") and encoding == "float":
                        p1 = prepare_inputs(clean[None], src[None], t, key, 1)
                        np.testing.assert_array_equal(
                            np.asarray(infer_dilation(model, p1, mode=mode)),
                            np.asarray(infer(model, p1)),
                        )
                        depth1_matches += 1
            path = out / f"{r['name']}-{mode}.npz"
            np.savez_compressed(path, **arrays)
            bindings[str(path)] = sha(path)
            rows.append(
                dict(
                    name=r["name"],
                    mode=mode,
                    outputs=len(arrays),
                    previous_mode_rms={
                        k: float(np.sqrt(np.mean((v - previous[k]) ** 2)))
                        for k, v in arrays.items()
                    }
                    if previous
                    else {},
                )
            )
            previous = arrays
        print(dict(case=r["name"], modes=3, outputs=12), flush=True)
    assert compared == 24 and depth1_matches == 12 and prior_matches == 12
    assert sum(x["outputs"] for x in rows) == 72
    assert states == {m: digest(nnx.state(model, nnx.Param)) for m, model in models.items()}
    write(
        out / "metrics.json",
        dict(
            rows=rows,
            outputs=72,
            states_unchanged=True,
            matched_existing_predictions=compared,
            depth1_matches=depth1_matches,
            prior_matches=prior_matches,
            matched_existing_exact=True,
            bindings=bindings,
            evaluation_seconds=time.perf_counter() - started,
        ),
    )
    render(root)


def paste(canvas, value, position):
    px = np.uint8(np.rint(np.clip((value[..., 0] + 1) * 127.5, 0, 255)))
    canvas.paste(Image.fromarray(px).convert("RGB"), position)


def render(root):
    plan = check(root)
    out = root / "evaluation"
    pages = []
    for r in plan["records"]:
        for encoding in ENCODINGS:
            if r["size"] == 256:
                canvas = Image.new("RGB", (768, 560), "white")
                pen = ImageDraw.Draw(canvas)
                bound = {}
                for col, mode in enumerate(MODES):
                    file = out / f"{r['name']}-{mode}.npz"
                    bound[str(file)] = sha(file)
                    with np.load(file) as z:
                        for row, m in enumerate(PATHS):
                            pen.text(
                                (col * 256 + 4, row * 280 + 4),
                                f"{r['name']} {m} {encoding} {mode}",
                                fill="black",
                            )
                            paste(canvas, z[f"{m}_{encoding}"], (col * 256, row * 280 + 24))
                file = out / f"native-{r['name']}-{encoding}.png"
                canvas.save(file)
                pages.append(dict(file=file.name, sha256=sha(file), predictions=bound))
            else:
                for m in PATHS:
                    canvas = Image.new("RGB", (1536, 540), "white")
                    pen = ImageDraw.Draw(canvas)
                    bound = {}
                    for item, mode in enumerate(MODES):
                        row, col = divmod(item, 3)
                        file = out / f"{r['name']}-{mode}.npz"
                        bound[str(file)] = sha(file)
                        with np.load(file) as z:
                            pen.text(
                                (col * 512 + 4, row * 540 + 4),
                                f"{r['name']} {m} {encoding} {mode}",
                                fill="black",
                            )
                            paste(canvas, z[f"{m}_{encoding}"], (col * 512, row * 540 + 24))
                    file = out / f"native-{r['name']}-{m}-{encoding}.png"
                    canvas.save(file)
                    pages.append(dict(file=file.name, sha256=sha(file), predictions=bound))
    assert len(pages) == 16
    write(
        out / "manifest.json",
        dict(records=pages, native_scale=True, thresholding=False, contrast_rescaling=False),
    )


def finalize(root):
    check(root)
    out = root / "evaluation"
    metrics = read(out / "metrics.json")
    assert metrics["outputs"] == 72 and metrics["states_unchanged"]
    assert metrics["matched_existing_exact"] and metrics["matched_existing_predictions"] == 24
    assert metrics["depth1_matches"] == 12 and metrics["prior_matches"] == 12
    for p, h in metrics["bindings"].items():
        assert sha(Path(p)) == h
    review = read(out / "visual-review.json")
    records = read(out / "manifest.json")["records"]
    assert review["reviewed"] and len(records) == len(review["observations"]) == 16
    for r in records:
        assert sha(out / r["file"]) == r["sha256"] == review["files"][r["file"]]
        assert r["file"] in review["observations"]
        for p, h in r["predictions"].items():
            assert sha(Path(p)) == h
    assert read(root / "driver-evaluate.json")["returncode"] == 0
    write(
        root / "completion.json",
        dict(
            stage="complete_scale_dilation",
            outputs=72,
            native_pages_reviewed=16,
            training=False,
            adoption=False,
            fresh_confirmation=False,
            evidence={
                f: sha(root / f)
                for f in (
                    "plan.json",
                    "evaluation/metrics.json",
                    "evaluation/manifest.json",
                    "evaluation/visual-review.json",
                    "diagnostics/decision.json",
                    "driver-evaluate.json",
                )
            },
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "evaluate", "finalize"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    globals()[args.action](args.root.resolve())
