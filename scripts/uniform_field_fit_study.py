"""Reuse the fixed FD fitting protocol, changing only the teacher width policy."""

import argparse
import json
from pathlib import Path

import field_fit_study as fit
import numpy as np
from campaign_protocol import read, sha, write
from field_repair_study import put
from PIL import Image, ImageDraw
from uniform_field_inputs import AUDITS, inputs

CONTROL = Path("../jax-field-fit-20261015/runs/field-fit-20261015").resolve()
fit.DOC = Path("docs/uniform-field-fit-plan-20261015.md").resolve()
fit.inputs = inputs


def prepare(root):
    assert read(CONTROL / "completion.json")["stage"] == "complete_field_fit"
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
        teacher_policy="all paths opaque2px; input unchanged", control=str(CONTROL), native_pages=12
    )
    write(root / "plan.json", p)


def train(root, until):
    fit.train(root, until)
    write(root / f"input-teacher-audit-{until}.json", AUDITS)


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
                "relative oracle": old["oracle"],
                "uniform oracle": new["oracle"],
                "initial": new["0"],
                "relative fit1200": old["1200"],
                "uniform fit1200": new["1200"],
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
        json.loads(x)
        for x in (CONTROL / "training/sampling.jsonl").read_text().splitlines()
    ]
    new_rows = [
        json.loads(x)
        for x in (root / "training/sampling.jsonl").read_text().splitlines()
    ]
    for a, b in zip(old_rows, new_rows, strict=True):
        for key in ("step", "case", "depth", "context_keep"):
            assert a[key] == b[key]
        if a["case"].startswith("probe"):
            assert a["input_target_mask_sha256"] == b["input_target_mask_sha256"]
    original.update(
        stage="complete_uniform_field_fit",
        native_pages_reviewed=12,
        input_policy_only_teacher_changed=True,
    )
    paths = [p for p in root.rglob("*.json") if p.name != "completion.json"]
    original["evidence"].update({str(p.relative_to(root)): sha(p) for p in paths})
    write(root / "completion.json", original)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("prepare", "train", "evaluate", "finalize"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--until", type=int, choices=(400, 800, 1200))
    args = p.parse_args()
    if args.phase == "train":
        train(args.root.resolve(), args.until)
    else:
        globals()[args.phase](args.root.resolve())
