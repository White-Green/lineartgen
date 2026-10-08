"""Native-pixel review sheets with fixed input/teacher-based crop selections."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
p.add_argument("--step", type=int, required=True)
a = p.parse_args()
r = a.root.resolve()


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


methods = read(r / "protocol.json")["methods"]
baseline = read(r / "protocol.json")["counterpart"]["method"]
labels = ["target", "source", "noisy"] + read(r / f"comparison-panels-{a.step}.json")["labels"]
ref = read(r / "evaluation.json")
casesroot = Path(ref["root"])
manifest = read(casesroot / "cases.json")
assert sha(casesroot / "cases.json") == ref["cases_sha256"]
out = r / f"visual-review/step-{a.step}"
out.mkdir(parents=True, exist_ok=True)
records = []
short = {
    **{
        n: n.replace("sqrt_ink", "sqrt").replace("strokes", "str").replace("mixed", "mix")
        for n in methods
    },
    "interval_control": "interval_ctrl",
    "interval_ink_guard": "interval_ink",
    "ink_guard": "hard_ink",
    "source_control": "guide_ctrl",
    "mixed_source_control": "guide_mix",
    "source_ink_guard": "source_ink",
    "source_interval_ink_guard": "source_intink",
}


def source(case, mode, steps):
    suffix = "-diagnostic" if steps > 1 else ""
    return (
        r
        / f"comparison-panels/evaluation/step-{a.step}-{mode}{suffix}"
        / f"case-{case:04d}-steps-{steps}-krita-white.png"
    )


def render(name, rows, columns, mode="ema"):
    canvas = Image.new("L", (128 * len(columns), 120 * len(rows)), 255)
    used = []
    for row, (case, rect) in enumerate(rows):
        for col, (label, steps) in enumerate(columns):
            path = source(case, mode, steps)
            with Image.open(path) as im:
                n = im.height - 24
                assert im.mode == "L" and im.width == n * len(labels)
                x0, y0, x1, y1 = rect
                assert x1 - x0 == 128 and y1 - y0 == 96
                assert 0 <= x0 < x1 <= n and 0 <= y0 < y1 <= n
                i = labels.index(label)
                canvas.paste(
                    im.crop((n * i + x0, y0 + 24, n * i + x1, y1 + 24)), (col * 128, row * 120 + 24)
                )
            ImageDraw.Draw(canvas).text(
                (col * 128 + 2, row * 120 + 2), f"{case} {short.get(label, label)} {steps}", fill=0
            )
            used.append(
                dict(
                    case=case,
                    rectangle=rect,
                    source=str(path),
                    sha256=sha(path),
                    column=label,
                    steps=steps,
                )
            )
    dest = out / name
    canvas.save(dest)
    records.append(dict(output=str(dest), sha256=sha(dest), inputs=used))


for mode in ["ema", "raw"]:
    for suite, ids in {
        "blur-05": [807, 903, 999, 1095, 1191, 1287, 1383, 1479],
        "strokes-01": [814, 910, 1006, 1102, 1198, 1294, 1390, 1486],
        "strokes-05": [815, 911, 1007, 1103, 1199, 1295, 1391, 1487],
    }.items():
        rows = []
        for i in ids:
            file = casesroot / manifest["cases"][i]["file"]
            assert sha(file) == ref["files"][manifest["cases"][i]["file"]]
            with np.load(file, allow_pickle=False) as d:
                x, y = np.median(d["probes"][:, :2], axis=0).round().astype(int)
            rows.append((i, (int(x - 64), int(y - 48), int(x + 64), int(y + 48))))
        render(
            f"{mode}-{suite}.png",
            rows,
            [(n, 1) for n in ["target", "source", "control", baseline, *methods]],
            mode,
        )
render(
    "real-ema.png",
    [(479, (230, 210, 358, 306)), (480, (230, 210, 358, 306)), (640, (224, 360, 352, 456))],
    [(n, 1) for n in ["target", "source", "control", baseline, *methods]],
)
render(
    "real-raw.png",
    [(480, (230, 210, 358, 306))],
    [(n, 1) for n in ["target", "source", "control", baseline, *methods]],
    "raw",
)
render(
    "repeat-one-five.png",
    [
        (0, (192, 176, 320, 272)),
        (640, (224, 360, 352, 456)),
        (1295, (192, 208, 320, 304)),
        (1391, (192, 208, 320, 304)),
    ],
    [("target", 1), ("source", 1)]
    + [(m, s) for m in ["control", baseline, *methods] for s in [1, 5]],
)
p = r / f"blank-evaluation/step-{a.step}-ema/case-0007-steps-5-krita-white.png"
with Image.open(p) as im:
    n = im.height - 24
    assert im.width == n * (3 + len(methods))
    canvas = Image.new("L", (n * len(methods), n + 24), 255)
    for i, _m in enumerate(methods):
        canvas.paste(im.crop(((i + 3) * n, 0, (i + 4) * n, n + 24)), (i * n, 0))
    dest = out / "blank-partial-five.png"
    canvas.save(dest)
records.append(
    dict(output=str(dest), sha256=sha(dest), inputs=[dict(source=str(p), sha256=sha(p))])
)
(out / "render.py").write_text(Path(__file__).read_text())
(out / "crops.json").write_text(
    json.dumps(
        dict(
            records=records,
            script_sha256=sha(out / "render.py"),
            display=(
                "Unchanged native uint8 Krita-white prediction pixels; no resizing. "
                "Float development input preparation, not the actual BGRA input API."
            ),
        ),
        indent=2,
    )
    + "\n"
)
print(out)
