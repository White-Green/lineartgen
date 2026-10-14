"""Independent oracle and learned-field rendering comparison; no training."""

import argparse
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from campaign_protocol import read, sha, write
from contour_coverage import contour_image
from flax import nnx
from opaque_geometry import render as render_geometry
from opaque_geometry_study import paste
from PIL import Image, ImageDraw
from subpixel_field_data import central_field, geometry_field
from subpixel_field_model import coverage, field_forward, load_field

from lineartgen.diffusion import pyramid

DQ = Path("../jax-opaque-geometry-20261014/runs/opaque-geometry-20261014").resolve()
DR = Path("../jax-subpixel-field-20261014/runs/subpixel-field-20261014").resolve()
DOC = Path("docs/contour-coverage-plan-20261014.md")
SEED = 2026101410


def prepare(root):
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    assert read(DQ / "completion.json")["stage"] == "complete_opaque_geometry_oracle"
    root.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(
        ["git", "archive", "--format=tar.gz", "-o", str(root / "source.tar.gz"), commit], check=True
    )
    write(
        root / "plan.json",
        dict(
            commit=commit,
            seed=SEED,
            oracle=read(DQ / "plan.json"),
            dq=str(DQ),
            dr=str(DR),
            document_sha256=sha(DOC),
            archive_sha256=sha(root / "source.tar.gz"),
            code={
                str(p): sha(p)
                for p in map(
                    Path,
                    [
                        "scripts/contour_coverage.py",
                        "scripts/contour_coverage_study.py",
                        "tests/test_contour_coverage.py",
                    ],
                )
            },
        ),
    )


def check(root):
    plan = read(root / "plan.json")
    assert sha(DOC) == plan["document_sha256"]
    for f, h in plan["code"].items():
        assert sha(Path(f)) == h
    for r in plan["oracle"]["records"]:
        assert sha(DQ / r["file"]) == r["sha256"]
    return plan


def mask(size, mode):
    yy, xx = np.indices((size, size))
    return (
        np.full((size, size, 1), 0.5, np.float32)
        if mode == "whole"
        else np.where((xx > size * 0.6 - yy * 0.25)[..., None], 0.5, 0).astype(np.float32)
    )


def oracle(root):
    plan = check(root)
    out = root / "oracle"
    out.mkdir(exist_ok=True)
    rows = []
    for r in plan["oracle"]["records"]:
        with np.load(DQ / r["file"]) as z:
            data = {k: z[k] for k in z.files}
        field = central_field(
            geometry_field(
                *(jnp.asarray(data[k]) for k in ("segments", "widths", "rectangles")),
                size=r["size"],
            )[None]
        )[..., None]
        fixed = np.asarray(coverage(field))[0]
        contour = np.asarray(contour_image(field))[0]
        same_backend = np.asarray(
            render_geometry(
                *(jnp.asarray(data[k]) for k in ("segments", "widths", "rectangles")),
                size=r["size"],
                supersampling=4,
            )
        )
        np.testing.assert_allclose(fixed, same_backend, atol=2e-6)
        with np.load(DQ / "outputs" / f"{r['name']}.npz") as z:
            dq_cross_backend_max_abs = float(abs(fixed - z["fine"]).max())
        assert np.isfinite(contour).all() and contour.min() >= -1 and contour.max() <= 1
        scales = [np.asarray(x)[0] for x in pyramid(jnp.asarray(contour)[None], 6)]
        masses = [float(((1 - x) / 2).sum()) * 4**i for i, x in enumerate(scales)]
        np.testing.assert_allclose(masses, masses[0], rtol=2e-5, atol=2e-3)
        t = mask(r["size"], "partial")
        partial = np.where(t > 0, contour, data["source"])
        np.testing.assert_array_equal(partial[t == 0], data["source"][t == 0])
        p = out / f"{r['name']}.npz"
        np.savez_compressed(p, fixed=fixed, contour=contour, partial=partial)
        a, b = (1 - fixed) / 2, (1 - contour) / 2
        rows.append(
            dict(
                name=r["name"],
                mass_by_level=masses,
                coverage_mae=float(abs(a - b).mean()),
                dq_cross_backend_max_abs=dq_cross_backend_max_abs,
                ink_relative_change=float((b.sum() - a.sum()) / max(float(a.sum()), 1)),
                sha256=sha(p),
                protected_exact=True,
            )
        )
        print(dict(oracle=r["name"]), flush=True)
    write(out / "metrics.json", dict(rows=rows, learned_model=False))
    render(root, False)


@nnx.jit
def predict_field(model, prepared):
    return field_forward(model, prepared)[0]


def evaluate(root):
    from cue_capacity import prepare_inputs

    plan = check(root)
    out = root / "learned"
    out.mkdir(exist_ok=True)
    paths = {m: DR / f"snapshots/step-6000/{m}/raw" for m in ("warm", "fresh")}
    binding = {str(p): sha(p) for folder in paths.values() for p in folder.iterdir() if p.is_file()}
    if (out / "selection.json").exists():
        assert read(out / "selection.json")["checkpoints"] == binding
    else:
        write(
            out / "selection.json",
            dict(
                checkpoints=binding,
                step=6000,
                models=["warm", "fresh"],
                seed=SEED,
                selection="predeclared both final raw models",
            ),
        )
    models = {m: load_field(p)[0] for m, p in paths.items()}
    rows = []
    for index, r in enumerate(plan["oracle"]["records"]):
        with np.load(DQ / r["file"]) as z:
            source, clean = z["source"], z["clean"]
        quantized = (
            np.rint(np.clip((source + 1) * 127.5, 0, 255)).astype(np.uint8).astype(np.float32)
            / 127.5
            - 1
        )
        for mode in ("whole", "partial"):
            t = mask(r["size"], mode)
            predictions = {}
            metrics = {}
            for encoding, src in (("float", source), ("uint8", quantized)):
                p = prepare_inputs(
                    clean[None],
                    src[None],
                    t[None],
                    jax.random.fold_in(jax.random.PRNGKey(SEED), index),
                    depth=6,
                )
                for m, model in models.items():
                    field = predict_field(model, p)
                    predictions[f"{m}_{encoding}_field"] = np.asarray(field)[0]
                    for decoder, fn in (("fixed", coverage), ("contour", contour_image)):
                        pred = np.where(t > 0, np.asarray(fn(field))[0], src)
                        assert np.isfinite(pred).all()
                        np.testing.assert_array_equal(pred[t == 0], src[t == 0])
                        predictions[f"{m}_{encoding}_{decoder}"] = pred
            for m in models:
                for decoder in ("fixed", "contour"):
                    metrics[f"{m}_{decoder}"] = float(
                        np.sqrt(
                            np.mean(
                                (
                                    predictions[f"{m}_float_{decoder}"]
                                    - predictions[f"{m}_uint8_{decoder}"]
                                )
                                ** 2
                            )
                        )
                    )
            path = out / f"{r['name']}-{mode}.npz"
            np.savez_compressed(path, **predictions)
            rows.append(
                dict(
                    name=r["name"],
                    mode=mode,
                    outputs=8,
                    protected_exact=True,
                    quantization_output_rms=metrics,
                    sha256=sha(path),
                )
            )
        print(dict(learned=r["name"]), flush=True)
    assert binding == {
        str(p): sha(p) for folder in paths.values() for p in folder.iterdir() if p.is_file()
    }
    write(
        out / "metrics.json",
        dict(rows=rows, outputs=sum(r["outputs"] for r in rows), checkpoints=binding),
    )
    render(root, True)


def render(root, learned):
    plan = check(root)
    out = root / ("learned" if learned else "oracle")
    pages = []
    labels = (
        ("uint8 rough source", "warm fixed", "warm contour", "fresh fixed", "fresh contour")
        if learned
        else ("rough source", "known geometry fixed", "known geometry contour")
    )
    for group, size, perpage in (("probe", 256, 4), ("scene", 512, 2)):
        records = [r for r in plan["oracle"]["records"] if r["name"].startswith(group)]
        for mode in ("whole", "partial") if learned else ("whole",):
            for first in range(0, len(records), perpage):
                selected = records[first : first + perpage]
                canvas = Image.new(
                    "RGB", (len(labels) * size, 34 + len(selected) * (size + 24)), "white"
                )
                draw = ImageDraw.Draw(canvas)
                for col, label in enumerate(labels):
                    draw.text((col * size + 6, 8), label, fill="black")
                for row, r in enumerate(selected):
                    with (
                        np.load(DQ / r["file"]) as inp,
                        np.load(
                            out / (f"{r['name']}-{mode}.npz" if learned else f"{r['name']}.npz")
                        ) as pred,
                    ):
                        source = inp["source"]
                        if learned:
                            source = (
                                np.rint(np.clip((source + 1) * 127.5, 0, 255))
                                .astype(np.uint8)
                                .astype(np.float32)
                                / 127.5
                                - 1
                            )
                        vals = [source] + (
                            [
                                pred[f"{m}_uint8_{d}"]
                                for m in ("warm", "fresh")
                                for d in ("fixed", "contour")
                            ]
                            if learned
                            else [pred["fixed"], pred["contour"]]
                        )
                        y = 34 + row * (size + 24)
                        draw.text((6, y), r["name"] + " / " + mode, fill="black")
                        for col, x in enumerate(vals):
                            paste(canvas, x, (col * size, y + 20))
                p = out / f"native-{group}-{mode}-{first}.png"
                canvas.save(p)
                pages.append(dict(file=p.name, sha256=sha(p), native=True))
    write(
        out / "manifest.json",
        dict(
            records=pages,
            learned_model=learned,
            final_raster_thresholding=False,
            contrast_rescaling=False,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "oracle", "evaluate"))
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    globals()[a.action](a.root.resolve())
