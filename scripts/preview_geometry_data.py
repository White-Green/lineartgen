"""Render independently seeded clean training shapes and their synthetic sketch inputs."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from PIL import Image, ImageDraw

from lineartgen.config import load_config
from lineartgen.data.geometry import KINDS, draw_geometry
from lineartgen.sketch import augment_sketch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    images, records = [], []
    for index, kind in enumerate(KINDS):
        key = [config.train.seed, index, 0x50524556]
        gray, info = draw_geometry(
            np.random.default_rng(np.random.SeedSequence(key)), 512, config.geometry, kind=kind
        )
        images.append(np.clip((gray * 2 - 1) * config.data.image_scale, -1, 1))
        records.append(dict(key=key, **info))
    clean = np.stack(images)
    settings = config.sketch.model_copy(update={"probability": 1.0})
    rough = np.asarray(
        jax.jit(lambda x: augment_sketch(x, jax.random.PRNGKey(705), settings))(clean)
    )
    for name, batch in (("clean", clean), ("rough", rough)):
        canvas = Image.new("L", (1536, 1608), 255)
        draw = ImageDraw.Draw(canvas)
        for index, gray in enumerate(batch):
            x, y = index % 3 * 512, index // 3 * 536
            draw.text((x + 8, y + 5), KINDS[index], fill=0)
            canvas.paste(
                Image.fromarray(
                    np.rint(np.clip((gray[..., 0] + 1) * 127.5, 0, 255)).astype(np.uint8)
                ),
                (x, y + 24),
            )
        canvas.save(args.output / f"{name}.png")
    np.savez_compressed(args.output / "preview.npz", clean=clean, rough=rough)
    (args.output / "seeds.json").write_text(
        json.dumps(
            dict(records=records, sketch_key=705, sketch_probability=1.0, preview_only=True),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
