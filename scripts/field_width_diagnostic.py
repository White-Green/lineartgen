"""Measure contour thickness without changing learned weights."""

import argparse
import subprocess
from pathlib import Path

import numpy as np
from campaign_protocol import read, sha, write
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, map_coordinates

PARENT = Path("../jax-aligned-recursion-20261016/runs/aligned-recursion-20261016").resolve()
DATA = Path("../jax-fresh-aligned-20261015/runs/fresh-aligned-20261015/data.json").resolve()
DOC = Path("docs/field-width-plan-20261016.md")
SS = 16
HALO = 8
DIAMETERS = (0.5, 1.0, 1.5)
MODES = ("current", "contour", "open0.5", "open1.0", "open1.5")


def opening(inside, diameter):
    radius = diameter * SS / 2
    padded = np.pad(inside, SS * 2)
    eroded = distance_transform_edt(padded) > radius
    if not eroded.any():
        return np.zeros_like(inside)
    opened = distance_transform_edt(~eroded) <= radius
    opened = opened[SS * 2 : -SS * 2, SS * 2 : -SS * 2]
    assert not (opened & ~inside).any()
    return opened


def area(inside):
    h, w = inside.shape
    assert h % SS == w % SS == 0
    return (1 - 2 * inside.reshape(h // SS, SS, w // SS, SS).mean((1, 3)))[..., None].astype(
        np.float32
    )


def sample(field, box):
    x0, y0, x1, y1 = box
    xs = (np.arange(x0 * SS, x1 * SS) + 0.5) / SS * 4 - 0.5
    ys = (np.arange(y0 * SS, y1 * SS) + 0.5) / SS * 4 - 0.5
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return map_coordinates(field[..., 0], [yy, xx], order=1, mode="nearest", prefilter=False)


def tests():
    y, x = np.indices((12 * SS, 12 * SS))
    broad = abs((y + 0.5) / SS - 6) < 1
    thin = abs((y + 0.5) / SS - 6) < 0.125
    for d in DIAMETERS:
        actual = opening(broad, d)
        np.testing.assert_array_equal(
            actual[3 * SS : 9 * SS, 3 * SS : 9 * SS], broad[3 * SS : 9 * SS, 3 * SS : 9 * SS]
        )
        np.testing.assert_array_equal(opening(actual, d), actual)
        assert not opening(thin, d).any()
    np.testing.assert_array_equal(area(np.zeros_like(thin)), np.ones((12, 12, 1)))
    np.testing.assert_array_equal(area(np.ones_like(thin)), -np.ones((12, 12, 1)))
    yy, xx = np.indices((48, 48))
    f = ((xx + 0.5) / 4 - 5.37)[..., None]
    a = area(sample(f, (0, 0, 12, 12)) < 0)
    expected = 1 - 2 * np.clip(5.37 - np.arange(12), 0, 1)
    err = float(np.max(abs(a[..., 0] - expected)) / 2)
    assert err <= 1 / 32 + 1e-7
    return dict(passed=True, band_tests=True, opening_subset_idempotent=True, plane_alpha_error=err)


def prepare(root):
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    assert read(PARENT / "completion.json")["stage"] == "complete_aligned_recursion"
    root.mkdir(parents=True, exist_ok=False)
    records = read(PARENT / "plan.json")["records"]
    bindings = {str(DATA): sha(DATA), str(DOC.resolve()): sha(DOC)}
    for r in records:
        bindings[r["input"]] = sha(Path(r["input"]))
        for enc in ("float", "uint8"):
            p = PARENT / "evaluation" / f"{r['name']}-{enc}-depth6.npz"
            bindings[str(p)] = sha(p)
    for p in (PARENT / "completion.json", PARENT / "plan.json"):
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
            bindings=bindings,
            source_archive_sha256=sha(root / "source.tar.gz"),
            outputs=50,
            native_pages=10,
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


def width_audit():
    records = [r for r in read(DATA)["records"] if r["split"] == "expanded"]
    assert len(records) == 512
    rows = []
    for r in records:
        paths = r["metadata"].get("path_geometry", [])
        peak = max((max(p["widths"]) for p in paths), default=1)
        lengths, widths, points = [], [], []
        for p in paths:
            w = np.asarray(p["widths"]) * (2 / peak)
            xy = np.asarray(p["points"])
            lengths.extend(np.linalg.norm(np.diff(xy, axis=0), axis=1))
            widths.extend((w[:-1] + w[1:]) / 2)
            points.extend(w)
        if not points:
            continue
        ll, ww, pp = np.array(lengths), np.array(widths), np.array(points)
        rows.append(
            dict(
                index=r["index"],
                kind=r["kind"],
                point_count=len(pp),
                min_width=float(pp.min()),
                max_width=float(pp.max()),
                length=float(ll.sum()),
                below={
                    str(d): dict(points=int((pp < d).sum()), length=float(ll[ww < d].sum()))
                    for d in DIAMETERS
                },
            )
        )
    return dict(
        records=512,
        path_records=len(rows),
        point_count=sum(r["point_count"] for r in rows),
        length=sum(r["length"] for r in rows),
        below={
            str(d): dict(
                points=sum(r["below"][str(d)]["points"] for r in rows),
                length=sum(r["below"][str(d)]["length"] for r in rows),
                records=sum(r["below"][str(d)]["points"] > 0 for r in rows),
            )
            for d in DIAMETERS
        },
        rows=rows,
    )


def pixels(a):
    return Image.fromarray(
        np.rint(np.clip((a[..., 0] + 1) * 127.5, 0, 255)).astype(np.uint8)
    ).convert("RGB")


def evaluate(root):
    plan = check(root)
    write(root / "tests.json", tests())
    write(root / "width-audit.json", width_audit())
    out = root / "evaluation"
    out.mkdir(exist_ok=False)
    rows, pages, bindings = [], [], {}
    for r in plan["records"]:
        box = r["crop"]
        padded = (box[0] - HALO, box[1] - HALO, box[2] + HALO, box[3] + HALO)
        trim = (slice(HALO, -HALO), slice(HALO, -HALO), slice(None))
        subtrim = (slice(HALO * SS, -HALO * SS), slice(HALO * SS, -HALO * SS))
        for enc in ("float", "uint8"):
            f = PARENT / "evaluation" / f"{r['name']}-{enc}-depth6.npz"
            with np.load(f) as z:
                field = z["normal_field"]
                current = z["normal"][box[1] : box[3], box[0] : box[2]]
            inside = sample(field, padded) < 0
            arrays = dict(current=current, contour=area(inside)[trim])
            removed = {}
            for d in DIAMETERS:
                opened = opening(inside, d)
                arrays[f"open{d:.1f}"] = area(opened)[trim]
                removed[str(d)] = float((inside & ~opened)[subtrim].sum() / SS**2)
            ff = field[4 * box[1] : 4 * box[3], 4 * box[0] : 4 * box[2], 0]
            gy, gx = np.gradient(ff, 0.25)
            boundary = np.zeros_like(ff, bool)
            boundary[1:] |= (ff[1:] < 0) != (ff[:-1] < 0)
            boundary[:-1] |= (ff[1:] < 0) != (ff[:-1] < 0)
            boundary[:, 1:] |= (ff[:, 1:] < 0) != (ff[:, :-1] < 0)
            boundary[:, :-1] |= (ff[:, 1:] < 0) != (ff[:, :-1] < 0)
            slope = np.hypot(gx, gy)[boundary]
            rows.append(
                dict(
                    case=r["name"],
                    encoding=enc,
                    zero_contour_ink_area=float(inside[subtrim].sum() / SS**2),
                    removed_ink_area=removed,
                    zero_crossing_gradient_quantiles=np.quantile(
                        slope, [0, 0.1, 0.5, 0.9, 1]
                    ).tolist(),
                    gradient_samples=len(slope),
                )
            )
            pred = out / f"{r['name']}-{enc}.npz"
            np.savez_compressed(pred, **arrays, field=ff)
            bindings[str(pred)] = sha(pred)
            w, h = box[2] - box[0], box[3] - box[1]
            stride = max(w, 130)
            canvas = Image.new("RGB", (stride * len(MODES), h + 30), "white")
            pen = ImageDraw.Draw(canvas)
            for col, mode in enumerate(MODES):
                pen.text((col * stride + 3, 4), mode, fill="black")
                canvas.paste(pixels(arrays[mode]), (col * stride, 25))
            image = out / f"detail-{r['name']}-{enc}.png"
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
        print(dict(case=r["name"], outputs=10), flush=True)
    write(out / "metrics.json", dict(rows=rows, outputs=50, bindings=bindings))
    write(out / "manifest.json", dict(records=pages, native_scale=True, diagnostic_only=True))


def finalize(root):
    check(root)
    m = read(root / "evaluation/metrics.json")
    assert read(root / "tests.json")["passed"] and m["outputs"] == 50
    for f, h in m["bindings"].items():
        assert sha(Path(f)) == h
    review = read(root / "evaluation/visual-review.json")
    records = read(root / "evaluation/manifest.json")["records"]
    assert review["reviewed"] and len(records) == len(review["observations"]) == 10
    for r in records:
        assert sha(root / "evaluation" / r["file"]) == r["sha256"] == review["files"][r["file"]]
        assert r["file"] in review["observations"]
    assert read(root / "driver-evaluate.json")["returncode"] == 0
    files = [
        "plan.json",
        "tests.json",
        "width-audit.json",
        "driver-evaluate.json",
        "evaluation/metrics.json",
        "evaluation/manifest.json",
        "evaluation/visual-review.json",
        "diagnostics/decision.json",
    ]
    write(
        root / "completion.json",
        dict(
            stage="complete_field_width",
            outputs=50,
            native_pages_reviewed=10,
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
