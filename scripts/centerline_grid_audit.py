"""HQ: geometry representation only; no learned prediction or fresh evaluation."""

import argparse
import datetime
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from campaign_protocol import read, sha, write
from centerline_grid import encode, infer_paths, render
from PIL import Image, ImageDraw

DQ = Path("../jax-opaque-geometry-20261014/runs/opaque-geometry-20261014").resolve()
CELLS = (0.5, 1.0, 2.0, 4.0)
DOC = Path("docs/centerline-grid-audit-plan-20261018.md")
CODE = [
    Path(p)
    for p in (
        "scripts/centerline_grid.py",
        "scripts/centerline_grid_audit.py",
        "tests/test_centerline_grid.py",
    )
]


def fixtures():
    result = []

    def add(name, paths, width=2.0, rectangles=(), components=None):
        segments, widths, ids = [], [], []
        for pid, points in enumerate(paths):
            points = np.asarray(points, np.float64)
            segments.extend(np.stack((points[:-1], points[1:]), axis=1))
            widths.extend([[width, width]] * (len(points) - 1))
            ids.extend([pid] * (len(points) - 1))
        result.append(
            dict(
                name=name,
                size=256,
                segments=np.asarray(segments).reshape(-1, 2, 2),
                widths=np.asarray(widths).reshape(-1, 2),
                path_ids=np.asarray(ids),
                rectangles=np.asarray(rectangles).reshape(-1, 4),
                expected_components=components,
            )
        )

    add("straight", [[[32.2, 128.2], [223.7, 128.2]]], components=1)
    add("diagonal", [[[32.2, 32.2], [223.7, 223.7]]], components=1)
    theta = np.linspace(-0.8, 0.8, 65)
    add(
        "curve", [np.stack((128 + 100 * np.sin(theta), 64 + 100 * np.cos(theta)), -1)], components=1
    )
    for gap in (0.25, 0.5, 1.0, 2.0):
        add(
            f"parallel-{gap}",
            [[[48.2, 128.1], [208.2, 128.1]], [[48.2, 128.1 + gap], [208.2, 128.1 + gap]]],
            width=0.2,
            components=2,
        )
    add("cross", [[[64.2, 64.2], [192.2, 192.2]], [[64.2, 192.2], [192.2, 64.2]]], components=1)
    add("tee", [[[64.2, 128.2], [192.2, 128.2]], [[128.2, 64.2], [128.2, 128.2]]], components=1)
    add("short", [[[128.1, 128.1], [128.3, 128.1]]], width=0.2, components=1)
    for gap in (0.25, 1.0):
        add(
            f"gap-{gap}",
            [[[48.2, 128.2], [128.1, 128.2]], [[128.1 + gap, 128.2], [208.2, 128.2]]],
            width=0.2,
            components=2,
        )
    add("taper", [[[32.2, 128.2], [223.7, 128.2]]], components=1)
    result[-1]["widths"] = np.array([[0.1, 3.0]])
    add("dot", [[[128.2, 128.2], [128.2, 128.2]]], components=1)
    add("blank", [], components=0)
    add("fill", [], rectangles=[[32, 32, 224, 224]], components=0)
    assert len(result) == 16
    return result


def prepare(root):
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    assert read(DQ / "completion.json")["stage"] == "complete_opaque_geometry_oracle"
    root.mkdir(parents=True, exist_ok=False)
    (root / "inputs").mkdir()
    records = []
    bindings = {
        str(DQ / "completion.json"): sha(DQ / "completion.json"),
        str(DQ / "plan.json"): sha(DQ / "plan.json"),
    }
    for record in read(DQ / "plan.json")["records"]:
        source = DQ / record["file"]
        reference = DQ / "outputs" / f"{record['name']}.npz"
        assert sha(source) == record["sha256"]
        bindings[str(source)] = sha(source)
        bindings[str(reference)] = sha(reference)
        with np.load(source) as z:
            data = {k: z[k] for k in ("segments", "widths", "rectangles", "source")}
        data["path_ids"] = infer_paths(data["segments"])
        path = root / "inputs" / f"{record['name']}.npz"
        np.savez_compressed(path, **data)
        records.append(
            dict(
                name=record["name"],
                size=record["size"],
                group=record["name"].split("-")[0],
                file=str(path.relative_to(root)),
                sha256=sha(path),
                reference=str(reference),
            )
        )
    for item in fixtures():
        path = root / "inputs" / f"fixture-{item['name']}.npz"
        np.savez_compressed(
            path, **{k: item[k] for k in ("segments", "widths", "rectangles", "path_ids")}
        )
        records.append(
            dict(
                name="fixture-" + item["name"],
                size=256,
                group="fixture",
                file=str(path.relative_to(root)),
                sha256=sha(path),
                expected_components=item["expected_components"],
            )
        )
    assert len(records) == 40
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(
        ["git", "archive", "--format=tar.gz", "-o", str(root / "source.tar.gz"), commit], check=True
    )
    write(
        root / "plan.json",
        dict(
            created_at=datetime.datetime.now(datetime.UTC).isoformat(),
            commit=commit,
            cells=CELLS,
            records=records,
            bindings=bindings,
            code={str(p): sha(p) for p in CODE},
            document_sha256=sha(DOC),
            archive_sha256=sha(root / "source.tar.gz"),
            learned_predictor=False,
            fresh_used=False,
            native_pages=12,
            adoption=False,
            reference_tolerance=dict(atol=1e-5, rtol=1e-5),
        ),
    )


def check(root):
    p = read(root / "plan.json")
    assert sha(DOC) == p["document_sha256"]
    assert sha(root / "source.tar.gz") == p["archive_sha256"]
    for f, h in p["bindings"].items():
        assert sha(Path(f)) == h
    for f, h in p["code"].items():
        assert sha(Path(f)) == h
    for r in p["records"]:
        assert sha(root / r["file"]) == r["sha256"]
    return p


def run(root):
    plan = check(root)
    rows = []
    outputs = root / "outputs"
    outputs.mkdir(exist_ok=False)
    started = datetime.datetime.now(datetime.UTC).isoformat()
    t = time.monotonic()
    for record in plan["records"]:
        with np.load(root / record["file"]) as z:
            data = {k: z[k] for k in z.files}
        ref = render(data["segments"], data["widths"], data["rectangles"], record["size"])
        arrays = {"reference": ref}
        metrics = []
        reference_match = None
        reference_error = None
        if "reference" in record:
            with np.load(record["reference"]) as z:
                old = z["fine"]
            reference_match = bool(np.allclose(ref, old, atol=1e-5, rtol=1e-5))
            reference_error = float(abs(ref - old).max())
        for cell in CELLS:
            graph = encode(data["segments"], data["widths"], cell, data["path_ids"])
            pred = render(
                graph["segments"], graph["segment_widths"], data["rectangles"], record["size"]
            )
            assert np.isfinite(pred).all() and pred.min() >= -1 and pred.max() <= 1
            key = f"cell{cell:g}"
            arrays[key] = pred
            for name in ("vertices", "widths", "edges", "cells"):
                arrays[key + "_" + name] = graph[name]
            alpha = (1 - pred) / 2
            target = (1 - ref) / 2
            metrics.append(
                dict(
                    cell=cell,
                    **graph["stats"],
                    absolute_coverage_error_per_ink=float(
                        abs(alpha - target).sum() / max(float(target.sum()), 1)
                    ),
                    ink_relative_error=float(
                        abs(alpha.sum() - target.sum()) / max(float(target.sum()), 1)
                    ),
                    expected_components=record.get("expected_components"),
                )
            )
        path = outputs / f"{record['name']}.npz"
        np.savez_compressed(path, **arrays)
        row = dict(
            name=record["name"],
            metrics=metrics,
            reference_match=reference_match,
            reference_max_error=reference_error,
            output_sha256=sha(path),
        )
        rows.append(row)
        print(
            json.dumps(
                dict(
                    name=record["name"],
                    reference_match=reference_match,
                    collisions=[m["noncontact_path_collisions"] for m in metrics],
                )
            ),
            flush=True,
        )
    write(
        root / "metrics.json",
        dict(
            rows=rows,
            started_at=started,
            finished_at=datetime.datetime.now(datetime.UTC).isoformat(),
            seconds=time.monotonic() - t,
            learned_predictor=False,
            reference_all_within_declared_tolerance=all(
                r["reference_match"] is not False for r in rows
            ),
        ),
    )
    native = root / "native"
    native.mkdir()
    pages = []
    for group in ("probe", "scene", "fixture"):
        records = [r for r in plan["records"] if r["group"] == group]
        perpage = 2 if group == "scene" else 4
        size = 512 if group == "scene" else 256
        for start in range(0, len(records), perpage):
            chosen = records[start : start + perpage]
            canvas = Image.new("RGB", (5 * size, 30 + len(chosen) * (size + 24)), "white")
            draw = ImageDraw.Draw(canvas)
            for col, label in enumerate(
                ["original opaque paths"] + [f"one vertex / {v:g}px cell" for v in CELLS]
            ):
                draw.text((col * size + 4, 7), label, fill="black")
            hashes = {}
            for row, record in enumerate(chosen):
                file = outputs / f"{record['name']}.npz"
                hashes[str(file.resolve())] = sha(file)
                with np.load(file) as z:
                    for col, key in enumerate(["reference"] + [f"cell{v:g}" for v in CELLS]):
                        image = np.round(np.clip((z[key][..., 0] + 1) * 127.5, 0, 255)).astype(
                            np.uint8
                        )
                        y = 30 + row * (size + 24)
                        draw.text((col * size + 4, y + 2), record["name"], fill="black")
                        canvas.paste(Image.fromarray(image).convert("RGB"), (col * size, y + 24))
            path = native / f"{group}-{start // perpage}.png"
            canvas.save(path)
            pages.append(dict(file=path.name, sha256=sha(path), bindings=hashes))
    assert len(pages) == 12
    write(native / "manifest.json", dict(records=pages, native_scale=True, learned_predictor=False))


def finalize(root):
    plan = check(root)
    metrics = read(root / "metrics.json")
    manifest = read(root / "native/manifest.json")
    notes = read(root / "native/observations.json")
    assert len(metrics["rows"]) == 40 and len(manifest["records"]) == len(notes) == 12
    for row in metrics["rows"]:
        assert sha(root / "outputs" / f"{row['name']}.npz") == row["output_sha256"]
    for row in manifest["records"]:
        assert notes[row["file"]] and sha(root / "native" / row["file"]) == row["sha256"]
        for f, h in row["bindings"].items():
            assert sha(Path(f)) == h
    assert read(root / "test-results.json")["passed"]
    files = [
        root / "plan.json",
        root / "metrics.json",
        root / "test-results.json",
        root / "native/manifest.json",
        root / "native/observations.json",
        root / "decision.json",
    ] + [root / "native" / r["file"] for r in manifest["records"]]
    write(
        root / "completion.json",
        dict(
            stage="complete_centerline_grid_representation_audit",
            frozen_commit=plan["commit"],
            cases=40,
            native_pages_reviewed=12,
            training_updates=0,
            learned_predictor=False,
            adoption=False,
            fresh_used=False,
            reference_tolerance_passed=metrics["reference_all_within_declared_tolerance"],
            evidence={str(p.relative_to(root)): sha(p) for p in files},
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "finalize"))
    parser.add_argument("--root", type=Path, required=True)
    a = parser.parse_args()
    globals()[a.action](a.root.resolve())


if __name__ == "__main__":
    main()
