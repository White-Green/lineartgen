"""Pixel-preserving inspection crops of fixed panels; no new model inference."""

import argparse
from pathlib import Path

import numpy as np
from campaign_protocol import read, sha, write
from PIL import Image, ImageDraw

from lineartgen.training.lineart_study import study_methods


def render(root, step):
    reference = read(root / "evaluation.json")
    cases = Path(reference["root"])
    manifest = read(cases / "cases.json")
    rectangles = {
        ("evaluation", 479): (230, 210, 358, 306),
        ("evaluation", 480): (230, 210, 358, 306),
        ("evaluation", 640): (224, 360, 352, 456),
    }
    for index in (903, 910, 911, 1294, 1295, 1390, 1391):
        with np.load(cases / manifest["cases"][index]["file"], allow_pickle=False) as arrays:
            x, y = np.median(arrays["probes"][:, :2], axis=0).round().astype(int)
        rectangles["supplemental-evaluation", index] = (x - 64, y - 48, x + 64, y + 48)
    folder = root / "visual-review" / f"step-{step}"
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for mode in ("ema", "raw"):
        for (suite, index), rectangle in rectangles.items():
            source = (
                root / suite / f"step-{step}-{mode}" / f"case-{index:04d}-steps-1-krita-white.png"
            )
            labels = ["target", "source", "noisy"]
            if suite == "supplemental-evaluation":
                labels.append("initial")
            labels.extend(study_methods(read(root / "protocol.json")))
            with Image.open(source) as sheet:
                size = sheet.height - 24
                assert sheet.width == size * len(labels)
                x0, y0, x1, y1 = rectangle
                assert 0 <= x0 < x1 <= size and 0 <= y0 < y1 <= size
                width, height = (x1 - x0) * 2, (y1 - y0) * 2
                selected = [label for label in labels if label != "noisy"]
                result = Image.new("L", (width * len(selected), height + 24), 255)
                for position, label in enumerate(selected):
                    column = labels.index(label)
                    crop = sheet.crop((column * size + x0, y0 + 24, column * size + x1, y1 + 24))
                    crop = crop.resize((width, height), Image.Resampling.NEAREST)
                    result.paste(crop, (position * width, 24))
                    ImageDraw.Draw(result).text((position * width + 4, 4), label, fill=0)
                output = folder / f"{mode}-case-{index:04d}.png"
                result.save(output)
            records.append(
                dict(
                    case=index,
                    mode=mode,
                    rectangle=[int(n) for n in rectangle],
                    source=str(source),
                    source_sha256=sha(source),
                    output=str(output),
                    output_sha256=sha(output),
                )
            )
    write(
        folder / "crops.json",
        dict(
            records=records,
            display="Existing uint8 Krita-white outputs, cropped and shown at 2x nearest-neighbor",
            selection="Fixed panels; real crops cover eyes/junction, geometry uses teacher probes",
            script_sha256=sha(Path(__file__)),
        ),
    )
    print(f"Rendered {len(records)} inspection crops")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    render(args.root, args.step)
