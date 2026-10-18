import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser(description="Render the five predeclared native detail crops")
parser.add_argument("--root", type=Path, required=True)
root = parser.parse_args().root.resolve()
out = root / "diagnostics/details"
out.mkdir(parents=True, exist_ok=False)
cases = [
    ("scene1", (116, 146, 236, 290), 1),
    ("scene2", (65, 160, 190, 235), 2),
    ("scene7", (292, 140, 390, 238), 7),
    ("probe37", (40, 88, 112, 161), None),
    ("oldtrain36", (45, 105, 165, 170), None),
]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


records = []
for name, box, index in cases:
    if index is not None:
        ip = root / f"scene512/input-{index}.npz"
        pp = root / f"scene512/predictions-{index}-whole-0.npz"
        with np.load(ip) as a, np.load(pp) as b:
            values = [
                a["quantized"],
                b["initial_uint8"],
                b["winner_uint8"],
                b["consistent_uint8"],
                b["consistent_float"],
            ]
        labels = [
            "source uint8",
            "HM opening6000 uint8",
            "HO control6000",
            "HO opening6000 uint8",
            "HO opening6000 float",
        ]
    else:
        ip = root / ("data/new_probe/037.npz" if name == "probe37" else "data/expanded/288.npz")
        pp = root / (
            "evaluation/predictions/new_probe-037-whole05-0.npz"
            if name == "probe37"
            else "evaluation/predictions/old_train-036-whole05-0.npz"
        )
        with np.load(ip) as a, np.load(pp) as b:
            values = [a["source"], b["initial"], b["winner"], b["consistent"]]
        labels = ["source", "HM opening6000", "HO control6000", "HO opening6000"]
    w, h = box[2] - box[0], box[3] - box[1]
    stride = max(w, 130)
    canvas = Image.new("RGB", (stride * len(values), h + 28), "white")
    pen = ImageDraw.Draw(canvas)
    for col, (value, label) in enumerate(zip(values, labels, strict=True)):
        gray = np.rint(np.clip((value[..., 0] + 1) * 127.5, 0, 255)).astype(np.uint8)
        canvas.paste(Image.fromarray(gray).crop(box).convert("RGB"), (col * stride, 28))
        pen.text((col * stride + 2, 7), label, fill="black")
    p = out / (name + ".png")
    canvas.save(p)
    records.append(
        dict(
            file=p.name,
            sha256=sha(p),
            crop=box,
            native_scale=True,
            bindings={str(ip): sha(ip), str(pp): sha(pp)},
        )
    )
(out / "manifest.json").write_text(json.dumps(dict(records=records), indent=2) + "\n")
print("five native detail crops saved")
