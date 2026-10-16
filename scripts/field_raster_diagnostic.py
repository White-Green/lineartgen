"""Render fixed learned fields through their zero contour without changing weights."""

import argparse
import subprocess
from pathlib import Path

import numpy as np
from campaign_protocol import read, sha, write
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates

PARENT = Path("../jax-aligned-recursion-20261016/runs/aligned-recursion-20261016").resolve()
DOC = Path("docs/field-raster-plan-20261016.md")
MODES = ("current", "sign4", "contour16")


def raster(field, *, refine=4, tile=32):
    field = np.asarray(field)[..., 0]
    h, w = field.shape
    assert h % 4 == w % 4 == 0
    samples = 4 * refine
    result = np.empty((h // 4, w // 4, 1), np.float32)
    xs = (np.arange(w * refine, dtype=np.float64) + 0.5) / refine - 0.5
    for start in range(0, h // 4, tile):
        end = min(start + tile, h // 4)
        ys = (np.arange(start * samples, end * samples, dtype=np.float64) + 0.5) / refine - 0.5
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        values = map_coordinates(field, [yy, xx], order=1, mode="nearest", prefilter=False)
        alpha = (values < 0).reshape(end - start, samples, w // 4, samples).mean((1, 3))
        result[start:end, :, 0] = 1 - 2 * alpha
    return result


def tests():
    h, w = 8, 12
    y, x = np.indices((h * 4, w * 4), dtype=np.float64)
    field = (x + 0.5) / 4 - 5.37
    rendered = raster(field[..., None])
    expected = 1 - 2 * np.clip(5.37 - np.arange(w), 0, 1)
    assert np.max(np.abs(rendered[..., 0] - expected)) <= 1 / 16 + 1e-7
    # Alpha area error <= half of one sample interval, i.e. 1/32.
    assert np.max(np.abs((1 - rendered[..., 0]) / 2 - (1 - expected) / 2)) <= 1 / 32 + 1e-7
    for scale in (0.1, 10):
        np.testing.assert_array_equal(raster((field * scale)[..., None]), rendered)
    diagonal = (x + y + 0.71)[..., None]
    for value, expected_value in ((0.05, 1), (-0.05, -1)):
        np.testing.assert_array_equal(
            raster(np.full_like(diagonal, value)), np.full((h, w, 1), expected_value, np.float32)
        )
    np.testing.assert_array_equal(raster(field[..., None], tile=1), rendered)
    signed = ((x + 0.5) / 4 - 5.37)[..., None]
    hard = 1 - 2 * (signed[..., 0] < 0).reshape(h, 4, w, 4).mean((1, 3))
    np.testing.assert_array_equal(raster(signed, refine=1)[..., 0], hard)
    return dict(
        passed=True,
        positive_scale_invariant=True,
        constant_signs_exact=True,
        plane_alpha_error=float(np.max(np.abs((1 - rendered[..., 0]) / 2 - (1 - expected) / 2))),
        alpha_error_bound=1 / 32,
        tiling_exact=True,
        sign4_exact=True,
    )


def prepare(root):
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    assert read(PARENT / "completion.json")["stage"] == "complete_aligned_recursion"
    root.mkdir(parents=True, exist_ok=False)
    records = read(PARENT / "plan.json")["records"]
    bindings = {}
    for r in records:
        bindings[r["input"]] = sha(Path(r["input"]))
        for encoding in ("float", "uint8"):
            f = PARENT / "evaluation" / f"{r['name']}-{encoding}-depth6.npz"
            bindings[str(f)] = sha(f)
    for f in (DOC.resolve(), PARENT / "plan.json", PARENT / "completion.json"):
        bindings[str(f)] = sha(f)
    for name in subprocess.check_output(["git", "ls-files"], text=True).splitlines():
        if name.endswith((".py", ".toml")) or name == "uv.lock":
            f = Path(name).resolve()
            bindings[str(f)] = sha(f)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(
        ["git", "archive", "--format=tar.gz", "-o", str(root / "source.tar.gz"), commit], check=True
    )
    write(
        root / "plan.json",
        dict(
            commit=commit,
            records=records,
            bindings=bindings,
            source_archive_sha256=sha(root / "source.tar.gz"),
            outputs=30,
            native_pages=15,
            training=False,
            adoption=False,
            fresh_confirmation=False,
        ),
    )


def check(root):
    p = read(root / "plan.json")
    for f, h in p["bindings"].items():
        assert sha(Path(f)) == h
    assert sha(root / "source.tar.gz") == p["source_archive_sha256"]
    return p


def pixels(a):
    return Image.fromarray(
        np.rint(np.clip((a[..., 0] + 1) * 127.5, 0, 255)).astype(np.uint8)
    ).convert("RGB")


def evaluate(root):
    plan = check(root)
    write(root / "tests.json", tests())
    out = root / "evaluation"
    out.mkdir(exist_ok=False)
    pages, rows, bindings = [], [], {}
    for r in plan["records"]:
        box = r["crop"]
        w, h = box[2] - box[0], box[3] - box[1]
        for encoding in ("float", "uint8"):
            f = PARENT / "evaluation" / f"{r['name']}-{encoding}-depth6.npz"
            with np.load(f) as z:
                field = z["normal_field"]
                current = z["normal"]
            size = r["size"]
            alpha = np.clip(0.5 - 4 * field, 0, 1).reshape(size, 4, size, 4, 1).mean((1, 3))
            np.testing.assert_allclose(1 - 2 * alpha, current, atol=1e-6, rtol=0)
            arrays = dict(current=current, sign4=raster(field, refine=1), contour16=raster(field))
            no_inside = field.reshape(size, 4, size, 4, 1).min((1, 3)) >= 0
            sl = (slice(box[1], box[3]), slice(box[0], box[2]), slice(None))
            positive_gray = no_inside[sl] & (alpha[sl] > 0)
            rows.append(
                dict(
                    case=r["name"],
                    encoding=encoding,
                    positive_only_gray_pixels=int(positive_gray.sum()),
                    maximum_alpha_without_negative_sample=float(
                        np.max(np.where(no_inside[sl], alpha[sl], 0))
                    ),
                    old_render_max_abs=float(np.max(np.abs((1 - 2 * alpha) - current))),
                )
            )
            pred = out / f"{r['name']}-{encoding}.npz"
            np.savez_compressed(pred, **arrays)
            bindings[str(pred)] = sha(pred)
            canvas = Image.new("RGB", (3 * max(w, 130), h + 30), "white")
            pen = ImageDraw.Draw(canvas)
            for col, mode in enumerate(MODES):
                pen.text((col * max(w, 130) + 3, 4), mode, fill="black")
                canvas.paste(pixels(arrays[mode]).crop(box), (col * max(w, 130), 25))
            image = out / f"detail-{r['name']}-{encoding}.png"
            canvas.save(image)
            pages.append(
                dict(
                    file=image.name,
                    sha256=sha(image),
                    predictions={str(pred): sha(pred)},
                    native_scale=True,
                    crop=box,
                )
            )
        with np.load(r["input"]) as z:
            source = z["source"]
        source = (
            np.rint(np.clip((source + 1) * 127.5, 0, 255)).astype(np.uint8).astype(np.float32)
            / 127.5
            - 1
        )
        canvas = Image.new("RGB", (size * 4, size + 30), "white")
        pen = ImageDraw.Draw(canvas)
        for col, (label, value) in enumerate((("source", source), *arrays.items())):
            pen.text((col * size + 4, 4), label, fill="black")
            canvas.paste(pixels(value), (col * size, 25))
        image = out / f"full-{r['name']}.png"
        canvas.save(image)
        pages.append(
            dict(
                file=image.name,
                sha256=sha(image),
                predictions={str(pred): sha(pred)},
                native_scale=True,
            )
        )
        print(dict(case=r["name"], outputs=6), flush=True)
    write(
        out / "metrics.json",
        dict(rows=rows, outputs=30, current_render_matches=True, bindings=bindings),
    )
    write(
        out / "manifest.json",
        dict(
            records=pages,
            final_image_thresholding=False,
            field_zero_contour=True,
            native_scale=True,
        ),
    )


def finalize(root):
    check(root)
    out = root / "evaluation"
    m = read(out / "metrics.json")
    assert (
        read(root / "tests.json")["passed"] and m["outputs"] == 30 and m["current_render_matches"]
    )
    for f, h in m["bindings"].items():
        assert sha(Path(f)) == h
    review = read(out / "visual-review.json")
    records = read(out / "manifest.json")["records"]
    assert review["reviewed"] and len(records) == len(review["observations"]) == 15
    for r in records:
        assert sha(out / r["file"]) == r["sha256"] == review["files"][r["file"]]
        assert r["file"] in review["observations"]
    assert read(root / "driver-evaluate.json")["returncode"] == 0
    files = [
        "plan.json",
        "tests.json",
        "driver-evaluate.json",
        "evaluation/metrics.json",
        "evaluation/manifest.json",
        "evaluation/visual-review.json",
        "diagnostics/decision.json",
    ]
    write(
        root / "completion.json",
        dict(
            stage="complete_field_raster",
            outputs=30,
            native_pages_reviewed=15,
            training=False,
            adoption=False,
            fresh_confirmation=False,
            evidence={f: sha(root / f) for f in files},
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "evaluate", "finalize"))
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    globals()[a.action](a.root.resolve())
