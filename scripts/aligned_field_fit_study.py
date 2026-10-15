"""Reuse the fixed fitting protocol, aligning teachers with smooth input warps."""

import argparse
import json
from pathlib import Path

import field_fit_study as fit
import numpy as np
from aligned_field_inputs import AUDITS, inputs
from campaign_protocol import read, sha, write
from field_repair_study import put
from PIL import Image, ImageDraw

CONTROL = Path("../jax-uniform-field-fit-20261015/runs/uniform-field-fit-20261015").resolve()
fit.DOC = Path("docs/aligned-field-fit-plan-20261015.md").resolve()
fit.inputs = inputs


def prepare(root):
    assert read(CONTROL / "completion.json")["stage"] == "complete_uniform_field_fit"
    fit.prepare(root)
    p = read(root / "plan.json")
    files = [
        CONTROL / "completion.json",
        CONTROL / "plan.json",
        CONTROL / "training/sampling.jsonl",
    ]
    files += list((CONTROL / "evaluation").glob("*.npz"))
    p["bindings"].update({str(f): sha(f) for f in files})
    p.update(
        teacher_policy="aligned vector paths opaque2px; input unchanged",
        control=str(CONTROL),
        native_pages=14,
    )
    write(root / "plan.json", p)


def train(root, until):
    review = read(root / "teacher-review.json")
    assert review["reviewed"] and len(review["observations"]) == 2
    for record in read(root / "teacher-preview/manifest.json")["records"]:
        assert sha(root / "teacher-preview" / record["file"]) == record["sha256"]
    fit.train(root, until)
    write(root / f"input-teacher-audit-{until}.json", AUDITS)


def audit(root):
    fit.check(root)
    p = read(fit.PARENT / "plan.json")
    _, _, source, _, _, oracle = inputs(p, "scene7-whole")
    with np.load(CONTROL / "evaluation/scene7-whole.npz") as z:
        np.testing.assert_array_equal(source, z["source"])
        old = z["oracle"]
    out = root / "teacher-preview"
    out.mkdir(exist_ok=False)
    records = []
    for kind, box in (("native", None), ("detail", (292, 140, 390, 238))):
        w, h = (512, 512) if box is None else (98, 98)
        cell = max(w + 12, 120)
        canvas = Image.new("RGB", (3 * cell, h + 28), "white")
        pen = ImageDraw.Draw(canvas)
        for col, (label, v) in enumerate(
            (("source", source), ("canonical teacher", old), ("aligned teacher", oracle))
        ):
            put(canvas, v, col * cell, 28, box)
            pen.text((col * cell + 3, 6), label, fill="black")
        file = out / (kind + ".png")
        canvas.save(file)
        records.append(dict(file=file.name, sha256=sha(file), native=True, box=box))
    write(
        out / "manifest.json", dict(records=records, thresholding=False, contrast_rescaling=False)
    )
    write(root / "input-teacher-audit-preview.json", AUDITS)


def evaluate(root):
    fit.evaluate(root)
    out = root / "comparison"
    out.mkdir(exist_ok=False)
    records = []
    for case in ("scene7-whole", "probe37-whole"):
        with (
            np.load(root / f"evaluation/{case}.npz") as new,
            np.load(CONTROL / f"evaluation/{case}.npz") as old,
        ):
            np.testing.assert_array_equal(new["source"], old["source"])
            np.testing.assert_array_equal(new["0"], old["0"])
            values = {
                "source": new["source"],
                "canonical oracle": old["oracle"],
                "aligned oracle": new["oracle"],
                "initial": new["0"],
                "canonical fit1200": old["1200"],
                "aligned fit1200": new["1200"],
            }
        box = (292, 140, 390, 238) if case.startswith("scene") else (40, 88, 112, 161)
        for kind, crop in (("native", None), ("detail", box)):
            w, h = (
                (values["source"].shape[1], values["source"].shape[0])
                if crop is None
                else (box[2] - box[0], box[3] - box[1])
            )
            cell = max(w + 12, 120)
            canvas = Image.new("RGB", (6 * cell, h + 28), "white")
            pen = ImageDraw.Draw(canvas)
            for col, (label, v) in enumerate(values.items()):
                put(canvas, v, col * cell, 28, crop)
                pen.text((col * cell + 4, 6), label, fill="black")
            file = out / f"{case}-{kind}.png"
            canvas.save(file)
            records.append(dict(file=file.name, sha256=sha(file), native=True, crop=crop))
    write(
        out / "manifest.json", dict(records=records, thresholding=False, contrast_rescaling=False)
    )
    write(root / "input-teacher-audit-evaluate.json", AUDITS)


def finalize(root):
    assert read(root / "driver-audit.json")["returncode"] == 0
    preview = read(root / "teacher-review.json")
    assert preview["reviewed"] and len(preview["observations"]) == 2
    for record in read(root / "teacher-preview/manifest.json")["records"]:
        assert preview["observations"][record["file"]]
        assert sha(root / "teacher-preview" / record["file"]) == record["sha256"]
    fit.finalize(root)
    original = read(root / "completion.json")
    write(root / "diagnostics/base-completion.json", original)
    notes = read(root / "comparison/review.json")
    records = read(root / "comparison/manifest.json")["records"]
    assert len(notes) == len(records) == 4
    for record in records:
        assert (
            notes[record["file"]] and sha(root / "comparison" / record["file"]) == record["sha256"]
        )
    for file in root.glob("input-teacher-audit-*.json"):
        assert all(r["prepared_inputs_exact"] for r in read(file))
    old_rows = [
        json.loads(x) for x in (CONTROL / "training/sampling.jsonl").read_text().splitlines()
    ]
    new_rows = [json.loads(x) for x in (root / "training/sampling.jsonl").read_text().splitlines()]
    for a, b in zip(old_rows, new_rows, strict=True):
        for key in ("step", "case", "depth", "context_keep"):
            assert a[key] == b[key]
        if a["case"].startswith("probe"):
            assert a["input_target_mask_sha256"] == b["input_target_mask_sha256"]
    original.update(
        stage="complete_aligned_field_fit",
        native_pages_reviewed=14,
        input_policy_only_teacher_changed=True,
    )
    paths = [p for p in root.rglob("*.json") if p.name != "completion.json"]
    original["evidence"].update({str(p.relative_to(root)): sha(p) for p in paths})
    write(root / "completion.json", original)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("prepare", "audit", "train", "evaluate", "finalize"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--until", type=int, choices=(400, 800, 1200))
    args = p.parse_args()
    if args.phase == "train":
        train(args.root.resolve(), args.until)
    else:
        globals()[args.phase](args.root.resolve())
