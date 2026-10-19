"""Native-resolution comparisons of an initial and a continued checkpoint."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def build(root, parent, step, baseline=0):
    out = root / "progress-reviews" / f"step-{step}"
    out.mkdir(parents=True, exist_ok=True)
    groups = {256: [], 512: []}
    for record in json.loads((parent / "plan.json").read_text())["records"]:
        with np.load(parent / record["file"]) as z:
            source = z["source"]
        with np.load(root / "reviews" / f"step-{baseline}" / (record["name"] + ".npz")) as z:
            initial = z["d3-whole"]
        with np.load(root / "reviews" / f"step-{step}" / (record["name"] + ".npz")) as z:
            final = z["d3-whole"]
        groups[source.shape[0]].append((record["name"], source, initial, final))
    manifest = []
    for size, rows in groups.items():
        count = 4 if size == 256 else 2
        for i in range(0, len(rows), count):
            page = rows[i : i + count]
            canvas = Image.new("RGB", (size * 3, (size + 28) * len(page)), "#ddd")
            draw = ImageDraw.Draw(canvas)
            for y, (name, *images) in enumerate(page):
                for x, (label, image) in enumerate(
                    zip(
                        ("source", f"total {16200 + baseline}", f"total {16200 + step}"),
                        images,
                        strict=True,
                    )
                ):
                    draw.text((x * size + 3, y * (size + 28) + 3), name + " " + label, fill="black")
                    gray = np.rint((image[..., 0] + 1) * 127.5).clip(0, 255).astype(np.uint8)
                    canvas.paste(
                        Image.fromarray(gray).convert("RGB"), (x * size, y * (size + 28) + 28)
                    )
            p = out / f"{size}-{i // count:02d}.png"
            canvas.save(p)
            manifest.append(dict(file=p.name, cases=[p[0] for p in page]))
    (out / "manifest.json").write_text(
        json.dumps(
            dict(step=step, baseline=baseline, rows=manifest, visually_reviewed=False), indent=2
        )
        + "\n"
    )
    print(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--baseline", type=int, default=0)
    build(**vars(p.parse_args()))
