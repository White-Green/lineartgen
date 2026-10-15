"""Audit hard separation eligibility against the unchanged input-support criterion."""

import argparse
import copy
import subprocess
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from campaign_protocol import read, sha, write
from merge_family_data import merge_geometry as legacy_geometry
from opaque_geometry import geometry, render
from PIL import Image, ImageDraw
from rough_family_data import STRESS, rough_source, stress_options
from support_merge_data import merge_allowed, merge_geometry

EQ = Path("../jax-joint-geometry-20261015/runs/joint-geometry-20261015").resolve()
ET = Path("../jax-context-dropout-20261015/runs/context-dropout-20261015").resolve()
DOC = Path("docs/stroke-supervision-plan-20261015.md").resolve()


def prepare(root):
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    assert read(ET / "completion.json")["stage"] == "complete_context_dropout"
    root.mkdir(parents=True, exist_ok=False)
    data = read(EQ / "data.json")
    train = [r for r in data["records"] if r["kind"] == "parallel"]
    probes = [r for r in data["evaluation"] if r["split"] == "new_probe" and 36 <= r["index"] <= 41]
    assert len(train) == 48 and len(probes) == 6
    files = [DOC, EQ / "data.json", EQ / "scene512/plan.json", ET / "completion.json"]
    files += [EQ / r["file"] for r in train + probes]
    for r in probes:
        name = f"new_probe-{r['index']:03d}-whole05-0.npz"
        files += [EQ / "evaluation/predictions" / name, ET / "evaluation/predictions" / name]
    for folder in (EQ / "snapshots/step-18000/mixed/raw", ET / "snapshots/step-6000/dropout/raw"):
        files += list(folder.iterdir())
    files += [
        Path(f).resolve()
        for f in subprocess.check_output(["git", "ls-files"], text=True).splitlines()
        if f.endswith((".py", ".toml")) or f == "uv.lock"
    ]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(
        ["git", "archive", "--format=tar.gz", "-o", str(root / "source.tar.gz"), commit], check=True
    )
    write(
        root / "plan.json",
        dict(
            commit=commit,
            train=train,
            probes=probes,
            bindings={str(p.resolve()): sha(p) for p in files},
            archive_sha256=sha(root / "source.tar.gz"),
            training=False,
            adoption=False,
            fresh_confirmation=False,
        ),
    )


def check(root):
    p = read(root / "plan.json")
    for file, h in p["bindings"].items():
        assert sha(Path(file)) == h, file
    assert sha(root / "source.tar.gz") == p["archive_sha256"]
    return p


def oracle(record, merged):
    metadata = copy.deepcopy(record["metadata"])
    if merged:
        a, b = metadata["path_geometry"]
        points = (np.asarray(a["points"]) + np.asarray(b["points"])) / 2
        metadata["path_geometry"] = [
            dict(kind="bezier", points=points.tolist(), widths=[2.0] * len(points))
        ]
        metadata["paths"] = 1
    image = np.asarray(render(*[jnp.asarray(x) for x in geometry(metadata)], size=256))
    assert image.shape == (256, 256, 1) and np.isfinite(image).all()
    assert image.min() >= -1 and image.max() <= 1
    return image


def put(canvas, value, x, y):
    canvas.paste(
        Image.fromarray(np.uint8(np.rint(np.clip((value[..., 0] + 1) * 127.5, 0, 255)))).convert(
            "RGB"
        ),
        (x, y),
    )


def widths(records):
    rows = []
    for r in records:
        paths = r["metadata"].get("path_geometry", [])
        peak = max((max(x["widths"]) for x in paths), default=1)
        rows.append(dict(index=r["index"], path_peaks=[2 * max(x["widths"]) / peak for x in paths]))
    return rows


def evaluate(root):
    plan = check(root)
    tick = time.monotonic()
    out = root / "evaluation"
    out.mkdir(exist_ok=False)
    rows = []
    examples = {}
    pages = []
    bindings = {}
    for r in plan["train"]:
        old, new = legacy_geometry(r), merge_geometry(r)
        if old["merge_eligible"]:
            for key in old:
                np.testing.assert_array_equal(old[key], new[key])
        with np.load(EQ / r["file"]) as z:
            mixture = z["variants"][[1, 25]].mean(0)
            base = z["source"]
        for condition in STRESS:
            source = (
                base
                if condition == "base"
                else rough_source(mixture, stress_options(condition, r["index"]))
            )
            a, b = merge_allowed(source, old), merge_allowed(source, new)
            assert not a or b
            row = dict(
                index=r["index"],
                condition=condition,
                separation=float(new["merge_separation"]),
                old=a,
                new=b,
            )
            rows.append(row)
            if b and not a and r["index"] not in examples:
                examples[r["index"]] = (r, condition, source, row)
    assert len(rows) == 288
    for r in plan["probes"]:
        with np.load(EQ / r["file"]) as z:
            source = z["source"]
        old, new = legacy_geometry(r), merge_geometry(r)
        allowed = dict(
            old=merge_allowed(source, old),
            new=merge_allowed(source, new),
            separation=float(new["merge_separation"]),
        )
        name = f"new_probe-{r['index']:03d}-whole05-0.npz"
        with np.load(EQ / "evaluation/predictions" / name) as z:
            eq = z["mixed"]
        with np.load(ET / "evaluation/predictions" / name) as z:
            et = z["dropout"]
        values = dict(
            source=source, original=oracle(r, False), merged_oracle=oracle(r, True), eq=eq, et=et
        )
        f = out / f"probe{r['index']}.npz"
        np.savez_compressed(f, **values)
        bindings[str(f)] = sha(f)
        canvas = Image.new("RGB", (1280, 300), "white")
        pen = ImageDraw.Draw(canvas)
        pen.text(
            (4, 4),
            f"probe{r['index']} sep={allowed['separation']:.3f} "
            f"old={allowed['old']} support={allowed['new']}",
            fill="black",
        )
        for col, (label, v) in enumerate(values.items()):
            pen.text((col * 256 + 4, 22), label, fill="black")
            put(canvas, v, col * 256, 44)
        file = out / f"native-probe{r['index']}.png"
        canvas.save(file)
        pages.append(
            dict(file=file.name, sha256=sha(file), prediction=str(f), prediction_sha256=sha(f))
        )
        rows.append(dict(index=r["index"], condition="probe-base", **allowed))
    selected = sorted(examples.values(), key=lambda x: (-x[3]["separation"], x[0]["index"]))[:6]
    for page in range((len(selected) + 2) // 3):
        group = selected[page * 3 : page * 3 + 3]
        canvas = Image.new("RGB", (768, len(group) * 300), "white")
        pen = ImageDraw.Draw(canvas)
        sources = []
        for row, (r, condition, source, info) in enumerate(group):
            values = dict(source=source, original=oracle(r, False), merged_oracle=oracle(r, True))
            f = out / f"train{r['index']}-{condition}.npz"
            np.savez_compressed(f, **values)
            bindings[str(f)] = sha(f)
            sources.append(dict(file=str(f), sha256=sha(f)))
            pen.text(
                (4, row * 300 + 4),
                f"train{r['index']} {condition} sep={info['separation']:.3f} "
                "old=False support=True",
                fill="black",
            )
            for col, (label, v) in enumerate(values.items()):
                pen.text((col * 256 + 4, row * 300 + 22), label, fill="black")
                put(canvas, v, col * 256, row * 300 + 44)
        file = out / f"native-training-{page}.png"
        canvas.save(file)
        pages.append(dict(file=file.name, sha256=sha(file), sources=sources))
    data = read(EQ / "data.json")
    train_widths = widths(data["records"])
    scene_widths = widths(read(EQ / "scene512/plan.json")["records"])
    assert all(v == 2 for r in train_widths for v in r["path_peaks"])
    write(
        out / "manifest.json",
        dict(records=pages, native_scale=True, thresholding=False, contrast_rescaling=False),
    )
    write(
        out / "metrics.json",
        dict(
            rows=rows,
            train_conditions=288,
            added_conditions=sum(r["new"] and not r["old"] for r in rows[:288]),
            counts={
                c: dict(
                    old=sum(r["old"] for r in rows[:288] if r["condition"] == c),
                    new=sum(r["new"] for r in rows[:288] if r["condition"] == c),
                )
                for c in STRESS
            },
            training_path_peaks=train_widths,
            scene_path_peaks=scene_widths,
            selected_training=[x[3] for x in selected],
            bindings=bindings,
            reused_model_predictions=12,
            seconds=time.monotonic() - tick,
        ),
    )
    check(root)
    print(
        dict(pages=len(pages), added_conditions=sum(r["new"] and not r["old"] for r in rows[:288])),
        flush=True,
    )


def finalize(root):
    check(root)
    out = root / "evaluation"
    metrics = read(out / "metrics.json")
    manifest = read(out / "manifest.json")
    review = read(out / "visual-review.json")
    assert len(review["observations"]) == len(manifest["records"]) and review["reviewed"]
    for r in manifest["records"]:
        assert sha(out / r["file"]) == r["sha256"] == review["files"][r["file"]]
        assert r["file"] in review["observations"]
    for file, h in metrics["bindings"].items():
        assert sha(Path(file)) == h
    assert read(root / "driver-evaluate.json")["returncode"] == 0
    write(
        root / "completion.json",
        dict(
            stage="complete_stroke_supervision",
            training=False,
            adoption=False,
            fresh_confirmation=False,
            native_pages_reviewed=len(manifest["records"]),
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
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "evaluate", "finalize"))
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    globals()[args.action](args.root.resolve())
