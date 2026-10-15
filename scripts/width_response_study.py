"""Width-only perturbations of six known development strokes, fixed model."""

import argparse
import copy
import subprocess
import time
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from cue_capacity import prepare_inputs
from flax import nnx
from PIL import Image, ImageDraw
from plausible_data import raster
from plausible_study import digest
from rough_family_data import BASE, rough_source
from subpixel_field_model import infer, load_field

EI = Path("../jax-optimized-broad-20261014/runs/optimized-broad-20261014").resolve()
MODEL = EI / "snapshots/step-18000/group/raw"
DOC = Path("docs/width-response-plan-20261015.md")
FACTORS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
SEED = 2026101504


def prepare(root):
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    assert read(EI / "completion.json")["stage"] == "complete_optimized_broad"
    root.mkdir(parents=True, exist_ok=False)
    records = [
        r
        for r in read(EI / "data.json")["evaluation"]
        if r["split"] == "new_probe" and r["index"] >= 58
    ]
    assert len(records) == 6
    bindings = {str(p.resolve()): sha(p) for p in [DOC, EI / "data.json", EI / "completion.json"]}
    for r in records:
        for p in [
            EI / r["file"],
            EI / f"evaluation/predictions/new_probe-{r['index']:03d}-whole05-0.npz",
        ]:
            bindings[str(p)] = sha(p)
    for p in MODEL.iterdir():
        if p.is_file():
            bindings[str(p)] = sha(p)
    for f in subprocess.check_output(["git", "ls-files"], text=True).splitlines():
        if f.endswith((".py", ".toml")) or f == "uv.lock":
            p = Path(f).resolve()
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
            factors=FACTORS,
            seed=SEED,
            bindings=bindings,
            archive_sha256=sha(root / "source.tar.gz"),
            outputs=36,
            native_pages=6,
            training=False,
            fresh_confirmation=False,
            adoption=False,
        ),
    )


def check(root):
    p = read(root / "plan.json")
    for f, h in p["bindings"].items():
        assert sha(Path(f)) == h, f
    assert sha(root / "source.tar.gz") == p["archive_sha256"]
    return p


def evaluate(root):
    plan = check(root)
    assert jax.devices()[0].platform == "cpu"
    model, _ = load_field(MODEL)
    before = digest(nnx.state(model, nnx.Param))
    out = root / "evaluation"
    out.mkdir(exist_ok=False)
    tick = time.perf_counter()
    rows = []
    pages = []
    bindings = {}
    for r in plan["records"]:
        i = r["index"]
        canvas = Image.new("RGB", (6 * 256, 2 * 280 + 30), "white")
        pen = ImageDraw.Draw(canvas)
        for col, factor in enumerate(FACTORS):
            metadata = copy.deepcopy(r["metadata"])
            for path in metadata["path_geometry"]:
                path["widths"] = (np.asarray(path["widths"]) * factor).tolist()
            direction = np.asarray(r["direction"])
            mixture = np.stack(
                [raster(metadata, direction, offset, 1) for offset in [-2.0, 2.0]]
            ).mean(0)
            source = rough_source(mixture, BASE)
            with np.load(EI / r["file"]) as original:
                clean = original["variants"][13]
                if factor == 1:
                    np.testing.assert_array_equal(source, original["source"])
            p = prepare_inputs(
                clean[None],
                source[None],
                np.full_like(source[None], 0.5),
                jax.random.PRNGKey(SEED),
                3,
            )
            pred = np.asarray(infer(model, p))[0]
            assert np.isfinite(pred).all() and pred.min() >= -1 and pred.max() <= 1
            if factor == 1:
                with np.load(EI / f"evaluation/predictions/new_probe-{i:03d}-whole05-0.npz") as old:
                    np.testing.assert_allclose(pred, old["group"], atol=1e-4, rtol=1e-4)
                    backend_difference = float(abs(pred - old["group"]).max())
            dest = out / f"wide-{i}-factor-{factor}.npz"
            np.savez_compressed(dest, source=source, prediction=pred)
            bindings[str(dest)] = sha(dest)
            width = max(metadata["path_geometry"][0]["widths"])
            rows.append(
                dict(
                    index=i,
                    factor=factor,
                    width=width,
                    identity_source_exact=factor == 1,
                    cpu_gpu_max_abs_difference=backend_difference if factor == 1 else None,
                )
            )
            for row, a in enumerate([source, pred]):
                pixels = np.uint8(np.rint(np.clip((a[..., 0] + 1) * 127.5, 0, 255)))
                canvas.paste(Image.fromarray(pixels).convert("RGB"), (col * 256, 30 + row * 280))
            pen.text((col * 256 + 4, 5), f"probe{i} x{factor} width{width:.2f}", fill="black")
            pen.text((col * 256 + 4, 290), "EI group6 raw18000 / d3", fill="black")
        dest = out / f"native-wide-{i}.png"
        canvas.save(dest)
        pages.append(dict(file=dest.name, sha256=sha(dest)))
        print(dict(case=i, outputs=len(rows)), flush=True)
    assert digest(nnx.state(model, nnx.Param)) == before
    write(out / "manifest.json", dict(records=pages))
    write(
        out / "metrics.json",
        dict(
            rows=rows,
            outputs=36,
            identity_sources_exact=6,
            cpu_gpu_comparison_atol=1e-4,
            cpu_gpu_comparison_rtol=1e-4,
            model_state_unchanged=True,
            seconds=time.perf_counter() - tick,
            bindings=bindings,
        ),
    )


def finalize(root):
    check(root)
    out = root / "evaluation"
    m = read(out / "metrics.json")
    r = read(out / "visual-review.json")
    assert m["outputs"] == 36 and m["identity_sources_exact"] == 6 and m["model_state_unchanged"]
    for f, h in m["bindings"].items():
        assert sha(Path(f)) == h
    manifest = read(out / "manifest.json")["records"]
    assert r["reviewed"] and len(r["observations"]) == len(manifest) == 6
    for page in manifest:
        assert sha(out / page["file"]) == page["sha256"] == r["files"][page["file"]]
        assert page["file"] in r["observations"]
    write(
        root / "completion.json",
        dict(
            stage="complete_width_response",
            outputs=36,
            native_pages_reviewed=6,
            training=False,
            adoption=False,
            fresh_confirmation=False,
            evidence={
                str(p.relative_to(root)): sha(p)
                for p in [
                    root / "plan.json",
                    out / "metrics.json",
                    out / "manifest.json",
                    out / "visual-review.json",
                ]
            },
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["prepare", "evaluate", "finalize"])
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    globals()[a.action](a.root.resolve())
