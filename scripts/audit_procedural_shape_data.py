"""Freeze procedural teacher recipes and render a native audit before learning."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from campaign_protocol import read, sha, write
from cue_preserving_data import MODES, strength_map
from PIL import Image, ImageDraw
from procedural_shape_data import record, render, selected, validate

DOC = Path("docs/procedural-shape-data-plan-20261017.md")
FU = Path("../jax-local-ambiguity-20261016/runs/local-ambiguity-20261016").resolve()
GJ = Path("../jax-bar-fill-fit-20261017/runs/bar-fill-fit-20261017").resolve()
GK = Path("../jax-bar-fill-transfer-20261017/runs/bar-fill-transfer-20261017").resolve()


def prepare(root):
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    assert read(GJ / "decision.json")["known_pair_fit_passed"]
    assert not read(GK / "decision.json")["transfer_gate_passed"]
    root.mkdir(parents=True, exist_ok=False)
    records = [record(s, i) for s in range(6000) if selected(s) for i in range(4)]
    assert len(records) == 6000
    assert (
        len(
            {
                json.dumps(
                    {k: v for k, v in r.items() if k not in ("step", "item", "depth")},
                    sort_keys=True,
                )
                for r in records
            }
        )
        == 6000
    )
    for r in records:
        validate(r)
        assert r == record(r["step"], r["item"])
    assert sum(r["filled"] for r in records) == 3000
    assert [sum(r["depth"] == d for r in records) for d in range(1, 7)] == [1000] * 6
    write(root / "recipes.json", records)
    canonical = dict(
        center=[63, 128], width=62, length=160, angle=0, sigma=1.2, opacity=0.4, filled=False
    )
    fill = dict(canonical, center=[128, 128], width=192, length=192, filled=True)
    with np.load(FU / "inputs.npz") as z:
        for i, recipe in enumerate((canonical, fill)):
            a = render(recipe)
            np.testing.assert_array_equal(a["source"], z["source"][i])
            np.testing.assert_array_equal(a["clean"], z["clean"][i])
    pages = []
    for step in range(18, 24):
        canvas = Image.new("RGB", (1024, 584), "white")
        pen = ImageDraw.Draw(canvas)
        arrays = []
        for item, mode in enumerate((0, 1, 2, 1)):
            rec = record(step, item)
            a = render(rec)
            b = render(rec)
            for name in a:
                np.testing.assert_array_equal(a[name], b[name])
            mask = strength_map(MODES[mode])
            assert mask.shape == a["clean"].shape
            active_target = np.where(mask > 0, a["clean"], a["source"])
            values = (a["source"], active_target)
            row, start_col = divmod(item, 2)
            for j, value in enumerate(values):
                col = start_col * 2 + j
                px = np.uint8(np.rint(np.clip((value[..., 0] + 1) * 127.5, 0, 255)))
                canvas.paste(Image.fromarray(px).convert("RGB"), (col * 256, row * 292 + 36))
                name = "source" if j == 0 else "active target"
                label = (
                    f"d{rec['depth']} "
                    + ("fill" if rec["filled"] else "bar")
                    + f" {MODES[mode]} / {name}"
                )
                pen.text((col * 256 + 3, row * 292 + 5), label, fill="black")
            arrays.append(a | dict(mask=mask))
        file = root / f"preview-depth-{step % 6 + 1}.png"
        canvas.save(file)
        np.savez_compressed(
            root / f"preview-depth-{step % 6 + 1}.npz",
            **{k: np.stack([a[k] for a in arrays]) for k in arrays[0]},
        )
        pages.append(dict(file=file.name, sha256=sha(file)))
    stats = dict(
        images=6000,
        bars=3000,
        fills=3000,
        batches=1500,
        batches_each_depth=250,
        min_width=min(r["width"] for r in records),
        max_width=max(r["width"] for r in records),
        rejections=sum(r["rejected"] for r in records),
        angle_kind_counts={str(k): sum(r["angle_kind"] == k for r in records) for k in range(4)},
        canonical_style_count=sum(r["canonical_style"] for r in records),
        canonical_FU_pixels_exact=True,
        rerender_exact=True,
        deterministic_records=True,
        all_geometry_contained=True,
        unique_full_recipes=6000,
    )
    write(root / "audit.json", stats)
    files = [
        DOC.resolve(),
        FU / "inputs.npz",
        GJ / "completion.json",
        GJ / "decision.json",
        GK / "completion.json",
        GK / "decision.json",
    ]
    files += [p for p in root.iterdir() if p.is_file()]
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
            bindings={str(p): sha(p) for p in files},
            archive_sha256=sha(root / "source.tar.gz"),
            pages=pages,
            training=False,
            adoption=False,
            fresh_confirmation=False,
        ),
    )
    print(stats, flush=True)


def finalize(root):
    p = read(root / "plan.json")
    for f, h in p["bindings"].items():
        assert sha(Path(f)) == h, f
    assert sha(root / "source.tar.gz") == p["archive_sha256"]
    notes = read(root / "observations.json")
    assert len(notes) == len(p["pages"]) == 6
    for r in p["pages"]:
        assert sha(root / r["file"]) == r["sha256"] and notes[r["file"]]
    assert read(root / "decision.json")["teacher_preview_passed"]
    names = ("plan.json", "audit.json", "observations.json", "decision.json", "test-results.json")
    write(
        root / "completion.json",
        dict(
            stage="complete_procedural_shape_data_preview",
            reviewed_examples=24,
            reviewed_pages=6,
            training=False,
            adoption=False,
            fresh_confirmation=False,
            evidence={f: sha(root / f) for f in names},
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("prepare", "finalize"))
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    globals()[a.phase](a.root.resolve())
