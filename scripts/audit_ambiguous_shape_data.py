"""Freeze procedural teacher recipes and render a native audit before learning."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from ambiguous_shape_data import permitted_outputs, record, render, selected, validate
from campaign_protocol import read, sha, write
from cue_preserving_data import MODES, strength_map
from PIL import Image, ImageDraw
from procedural_shape_data import selected as long_selected

DOC = Path("docs/ambiguous-shape-data-plan-20261017.md")
FU = Path("../jax-local-ambiguity-20261016/runs/local-ambiguity-20261016").resolve()
GJ = Path("../jax-bar-fill-fit-20261017/runs/bar-fill-fit-20261017").resolve()
GK = Path("../jax-bar-fill-transfer-20261017/runs/bar-fill-transfer-20261017").resolve()


def prepare(root):
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    assert not read(GK / "decision.json")["transfer_gate_passed"]
    root.mkdir(parents=True, exist_ok=False)
    records = [record(s, i) for s in range(6000) if selected(s) for i in range(4)]
    assert len(records) == 3000
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
        == 3000
    )
    for r in records:
        validate(r)
        assert r == record(r["step"], r["item"])
    assert not any(long_selected(r["step"]) for r in records)
    assert [sum(r["depth"] == d for r in records) for d in range(1, 7)] == [500] * 6
    write(root / "recipes.json", records)
    pages, allowed_count, permitted_ink = [], 0, []
    for step in range(30, 36):
        canvas = Image.new("RGB", (768, 1168), "white")
        pen = ImageDraw.Draw(canvas)
        arrays = []
        for item, mode in enumerate((0, 1, 2, 1)):
            rec = record(step, item)
            a, b = render(rec), render(rec)
            for name in a:
                np.testing.assert_array_equal(a[name], b[name])
            mask = strength_map(MODES[mode])
            original, other, allowed = permitted_outputs(a, mask)
            assert allowed == (mode != 2)
            allowed_count += int(allowed)
            for value in (original, other):
                np.testing.assert_array_equal(value[mask == 0], a["source"][mask == 0])
            if allowed:
                ink = [float(((1 - value[mask > 0]) / 2).sum()) for value in (original, other)]
                assert min(ink) > 0
                permitted_ink.append(dict(step=step, item=item, ink=ink))
            else:
                np.testing.assert_array_equal(original, other)
            for col, (label, value) in enumerate(
                zip(
                    (
                        "source",
                        "primary allowed",
                        "other allowed" if allowed else "original only: partial",
                    ),
                    (a["source"], original, other),
                    strict=True,
                )
            ):
                pixels = np.uint8(np.rint(np.clip((value[..., 0] + 1) * 127.5, 0, 255)))
                canvas.paste(Image.fromarray(pixels).convert("RGB"), (col * 256, item * 292 + 36))
                pen.text(
                    (col * 256 + 3, item * 292 + 5),
                    f"d{rec['depth']} {MODES[mode]} / {label}",
                    fill="black",
                )
            arrays.append(a | dict(mask=mask, allowed=np.asarray(allowed)))
        file = root / f"preview-depth-{step % 6 + 1}.png"
        canvas.save(file)
        np.savez_compressed(
            root / f"preview-depth-{step % 6 + 1}.npz",
            **{k: np.stack([a[k] for a in arrays]) for k in arrays[0]},
        )
        pages.append(dict(file=file.name, sha256=sha(file)))
    assert allowed_count == 18
    stats = dict(
        images=3000,
        batches=750,
        batches_each_depth=125,
        min_width=min(r["width"] for r in records),
        max_width=max(r["width"] for r in records),
        min_ratio=min(r["width"] / r["length"] for r in records),
        max_ratio=max(r["width"] / r["length"] for r in records),
        angle_kind_counts={str(k): sum(r["angle_kind"] == k for r in records) for k in range(4)},
        square_recipes=sum(r["square"] for r in records),
        canonical_style_count=sum(r["canonical_style"] for r in records),
        rerender_exact=True,
        deterministic_records=True,
        all_geometry_contained=True,
        unique_full_recipes=3000,
        disjoint_from_long_bar_fill_batches=True,
        preview_alternate_allowed=18,
        preview_partial_original_only=6,
        both_permitted_axes_have_active_ink=permitted_ink,
        protected_pixels_exact=True,
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
            stage="complete_ambiguous_shape_data_preview",
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
