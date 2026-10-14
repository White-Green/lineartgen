"""Geometry rendering oracle only: no learned predictor, no product adoption."""

import argparse
import copy
import subprocess
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from campaign_protocol import read, sha, write
from opaque_geometry import compose, geometry, render
from PIL import Image, ImageDraw
from plausible_data import raster
from wide_style_data import allstroke_candidates

from lineartgen.data.geometry import normals
from lineartgen.diffusion import pyramid

PARENT = Path("../jax-coherent-path-20261014/runs/coherent-path-20261014")
CASES = (0, 6, 12, 18, 24, 30, 36, 38, 42, 48, 54, 56, 58, 59, 62, 63)
DOCUMENT = Path("docs/opaque-geometry-plan-20261014.md")


def prepare(root):
    assert read(PARENT / "completion.json")["stage"] == "complete_coherent_path"
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    root.mkdir(parents=True, exist_ok=False)
    (root / "inputs").mkdir()
    rows = {
        r["index"]: r for r in read(PARENT / "data.json")["evaluation"] if r["split"] == "new_probe"
    }
    records = []
    for i in CASES:
        row = rows[i]
        with np.load(PARENT / row["file"]) as z:
            source = z["source"]
            clean = allstroke_candidates(z, row)[13]
        s, w, r = geometry(row["metadata"], kind=row["kind"])
        p = root / f"inputs/probe-{i:02d}.npz"
        np.savez_compressed(p, source=source, clean=clean, segments=s, widths=w, rectangles=r)
        records.append(
            dict(
                name=f"probe-{i:02d}",
                index=i,
                kind=row["kind"],
                size=256,
                file=str(p.relative_to(root)),
                sha256=sha(p),
            )
        )
    for row in read(PARENT / "scene512/plan.json")["records"]:
        i = row["index"]
        metadata = copy.deepcopy(row["metadata"])
        peak = max(max(p["widths"]) for p in metadata["path_geometry"])
        for p in metadata["path_geometry"]:
            p["widths"] = (np.asarray(p["widths"]) * 2 / peak).tolist()
        direction = normals(np.asarray(metadata["path_geometry"][0]["points"]))[64]
        clean = raster(metadata, direction, 0, 1, size=512)
        with np.load(PARENT / f"scene512/input-{i}.npz") as z:
            source = z["quantized"]
        s, w, r = geometry(row["metadata"], size=512)
        p = root / f"inputs/scene-{i}.npz"
        np.savez_compressed(p, source=source, clean=clean, segments=s, widths=w, rectangles=r)
        records.append(
            dict(
                name=f"scene-{i}",
                index=i,
                kind="multi",
                size=512,
                file=str(p.relative_to(root)),
                sha256=sha(p),
            )
        )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(
        ["git", "archive", "--format=tar.gz", "-o", str(root / "source.tar.gz"), commit], check=True
    )
    write(
        root / "plan.json",
        dict(
            commit=commit,
            records=records,
            oracle=True,
            learned_model=False,
            adoption=False,
            document_sha256=sha(DOCUMENT),
            code={
                str(p): sha(p)
                for p in map(
                    Path,
                    [
                        "scripts/opaque_geometry.py",
                        "scripts/opaque_geometry_study.py",
                        "tests/test_opaque_geometry.py",
                    ],
                )
            },
            archive_sha256=sha(root / "source.tar.gz"),
        ),
    )


def check(root):
    p = read(root / "plan.json")
    assert sha(DOCUMENT) == p["document_sha256"]
    for f, h in p["code"].items():
        assert sha(Path(f)) == h
    for r in p["records"]:
        assert sha(root / r["file"]) == r["sha256"]
    return p


def run(root):
    plan = check(root)
    out = root / "outputs"
    out.mkdir(exist_ok=True)
    rows = []
    for record in plan["records"]:
        with np.load(root / record["file"]) as z:
            data = {k: z[k] for k in z.files}
        args = tuple(jnp.asarray(data[k]) for k in ("segments", "widths", "rectangles"))
        image = render(*args, size=record["size"], supersampling=4)
        fine = np.asarray(image)
        assert np.isfinite(fine).all() and fine.min() >= -1 and fine.max() <= 1
        levels = tuple(np.asarray(x)[0] for x in pyramid(image[None], 6))
        masses = [float(((1 - x) / 2).sum()) * 4**i for i, x in enumerate(levels)]
        np.testing.assert_allclose(masses, masses[0], rtol=2e-5, atol=2e-3)
        arrays = dict(fine=fine, **{f"level{i}": x for i, x in enumerate(levels)})
        stats = {}
        if record["size"] == 256:
            finer = np.asarray(render(*args, size=256, supersampling=8))
            arrays["reference8"] = finer
            aa, bb = (1 - fine) / 2, (1 - finer) / 2
            stats = dict(
                supersampling_mae=float(abs(aa - bb).mean()),
                supersampling_ink_relative=float(
                    abs(aa.sum() - bb.sum()) / max(float(bb.sum()), 1)
                ),
            )
            assert stats["supersampling_ink_relative"] < 0.015
        yy, xx = np.indices(fine.shape[:2])
        t = np.where((xx > record["size"] * 0.6 - yy * 0.25)[..., None], 0.5, 0).astype(np.float32)
        partial = np.asarray(compose(image, jnp.asarray(data["source"]), jnp.asarray(t)))
        np.testing.assert_array_equal(partial[t == 0], data["source"][t == 0])
        arrays["partial"] = partial
        path = out / f"{record['name']}.npz"
        np.savez_compressed(path, **arrays)
        rows.append(
            dict(
                name=record["name"],
                kind=record["kind"],
                mass_by_level=masses,
                protected_exact=True,
                output_sha256=sha(path),
                **stats,
            )
        )
        print(dict(case=record["name"], **stats), flush=True)
    write(root / "metrics.json", dict(rows=rows, oracle=True, learned_model=False))
    pages = []
    for group in ("probe", "scene"):
        records = [r for r in plan["records"] if r["name"].startswith(group)]
        perpage = 4 if group == "probe" else 2
        size = 256 if group == "probe" else 512
        labels = (
            ("rough source", "legacy polygon 2px", "opaque geometry 4x", "opaque geometry 8x")
            if group == "probe"
            else ("rough uint8 source", "legacy polygon 2px", "opaque geometry 4x")
        )
        for first in range(0, len(records), perpage):
            selected = records[first : first + perpage]
            canvas = Image.new(
                "RGB", (len(labels) * size, 34 + len(selected) * (size + 24)), "white"
            )
            draw = ImageDraw.Draw(canvas)
            for col, label in enumerate(labels):
                draw.text((col * size + 6, 8), label, fill="black")
            for row, r in enumerate(selected):
                with np.load(root / r["file"]) as inp, np.load(out / f"{r['name']}.npz") as pred:
                    values = [inp["source"], inp["clean"], pred["fine"]]
                    if group == "probe":
                        values.append(pred["reference8"])
                    y = 34 + row * (size + 24)
                    draw.text((6, y), r["name"] + " / known geometry, not prediction", fill="black")
                    for col, x in enumerate(values):
                        paste(canvas, x, (col * size, y + 20))
            file = root / f"native-{group}-{first}.png"
            canvas.save(file)
            pages.append(dict(file=file.name, sha256=sha(file), native=True))
    canvas = Image.new("RGB", (6 * 256, 34 + 4 * 280), "white")
    draw = ImageDraw.Draw(canvas)
    for level in range(6):
        draw.text(
            (level * 256 + 6, 8),
            f"level{level} / {256 // 2**level}px / nearest display",
            fill="black",
        )
    for row, i in enumerate((6, 30, 48, 56)):
        with np.load(out / f"probe-{i:02d}.npz") as z:
            for level in range(6):
                x = z[f"level{level}"]
                x = np.repeat(np.repeat(x, 2**level, 0), 2**level, 1)
                paste(canvas, x, (level * 256, 54 + row * 280))
        draw.text(
            (6, 34 + row * 280),
            f"probe-{i:02d} / same full-resolution coverage pyramid",
            fill="black",
        )
    file = root / "pyramid-display.png"
    canvas.save(file)
    pages.append(dict(file=file.name, sha256=sha(file), native=False))
    write(root / "manifest.json", dict(records=pages, thresholding=False, contrast_rescaling=False))


def paste(canvas, value, position):
    px = np.uint8(np.rint(np.clip((value[..., 0] + 1) * 127.5, 0, 255)))
    canvas.paste(Image.fromarray(px).convert("RGB"), position)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    globals()[args.action](args.root.resolve())
