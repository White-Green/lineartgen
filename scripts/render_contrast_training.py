"""Four deterministic training examples with intersecting unequal-opacity strokes."""

import argparse
from pathlib import Path

import numpy as np
from campaign_protocol import sha, write
from continuous_width_data import width_choice, width_source
from contrast_composition_data import overlay_source
from contrast_composition_study import SEED, load_data, select
from PIL import Image, ImageDraw
from subpixel_field_data import central_field
from subpixel_field_model import coverage

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
root = p.parse_args().root.resolve()
data = load_data(root)
examples = []
for step in range(100):
    s = select(step)
    for item, (i, opts, overlay) in enumerate(
        zip(s["indices"], s["augmentation"], s["overlays"], strict=True)
    ):
        ratio = overlay["augmentation"]["opacity"] / opts["opacity"]
        if not overlay["enabled"] or 0.55 < ratio < 1.8:
            continue
        j = overlay["index"]
        first = width_source(data, data["records"][i], i, width_choice(SEED, step, item, i), opts)
        second = width_source(
            data, data["records"][j], j, overlay["width"], overlay["augmentation"]
        )
        overlap = int(((first < 0.96) & (second < 0.96)).sum())
        if overlap < 8:
            continue
        field = np.minimum(data["fields"][i], data["fields"][j]).astype(np.float32)
        oracle = np.asarray(coverage(central_field(field[None])[..., None]))[0]
        examples.append(
            (
                first,
                second,
                overlay_source(first, second),
                oracle,
                dict(
                    step=step,
                    item=item,
                    primary=i,
                    secondary=j,
                    opacity_ratio=ratio,
                    overlapping_pixels=overlap,
                ),
            )
        )
        if len(examples) == 4:
            break
    if len(examples) == 4:
        break
assert len(examples) == 4
out = root / "diagnostics"
out.mkdir(exist_ok=True)
canvas = Image.new("RGB", (1024, 1160), "white")
pen = ImageDraw.Draw(canvas)
for col, label in enumerate(
    ("primary rough", "independent rough", "transparent overlay", "opaque geometry union")
):
    pen.text((col * 256 + 5, 5), label, fill="black")
for row, values in enumerate(examples):
    meta = values[-1]
    pen.text((5, row * 280 + 25), str(meta), fill="black")
    for col, value in enumerate(values[:4]):
        gray = np.rint(np.clip((value[..., 0] + 1) / 2, 0, 1) * 255).astype(np.uint8)
        canvas.paste(Image.fromarray(gray).convert("RGB"), (col * 256, row * 280 + 45))
target = out / "training-overlay-oracle.png"
canvas.save(target)
write(
    out / "training-overlay-oracle.json",
    dict(
        records=[x[-1] for x in examples],
        sha256=sha(target),
        native_size=256,
        thresholding=False,
        contrast_rescaling=False,
    ),
)
