"""Assemble sampling report figures from verified inference pixels without rescaling."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from summarize_training_methods import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--variant", default="low80")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []

    def verified(method, mode, case):
        folder = args.root / "evaluation" / f"widths-10000-{method}-{mode}"
        path = folder / "verification.json"
        audit = json.loads(path.read_text())
        assert audit["method"] == method and audit["mode"] == mode
        assert audit["weights_sha256"] == sha(
            args.root / f"snapshots/step-10000/{method}/{mode}/weights.npz"
        )
        assert audit["metrics_sha256"] == sha(
            args.root / f"evaluation/step-10000-{mode}/metrics.json"
        )
        row = next(r for r in audit["records"] if r["case"] == case)
        image = Path(row["image"])
        assert row["sha256"] == sha(image) and row["recursion_levels"] == 3
        with Image.open(image) as opened:
            pixels = np.asarray(opened).copy()
        assert pixels.shape == (536, 3072)
        return image, pixels[24:], {str(image): sha(image), str(path): sha(path)}

    for case, mode in ((256, "raw"), (480, "raw"), (480, "ema"), (479, "ema")):
        _, a, sources = verified("control", mode, case)
        _, b, other = verified(args.variant, mode, case)
        np.testing.assert_array_equal(a[:, :1536], b[:, :1536])
        sources.update(other)
        pixels = np.concatenate((a[:, :2048], b[:, 1536:2048]), axis=1)
        canvas = Image.new("L", (2560, 536), 255)
        canvas.paste(Image.fromarray(pixels), (0, 24))
        draw = ImageDraw.Draw(canvas)
        for i, label in enumerate(
            (
                "target (preserve mask)",
                "rough source",
                "noisy input",
                f"S uniform {mode}",
                f"S {args.variant} {mode}",
            )
        ):
            draw.text((i * 512 + 8, 5), label, fill=0)
        output = args.output / f"paired-{mode}-case-{case:03d}.png"
        canvas.save(output)
        records.append(
            dict(
                case=case,
                mode=mode,
                output=str(output),
                sha256=sha(output),
                source_sha256=sources,
                inputs_exact=True,
            )
        )
    for mode in ("raw", "ema"):
        source, _, sources = verified(args.variant, mode, 1376)
        output = args.output / f"{args.variant}-{mode}-widths-case-1376.png"
        shutil.copyfile(source, output)
        assert sha(source) == sha(output)
        records.append(
            dict(
                case=1376,
                mode=mode,
                output=str(output),
                sha256=sha(output),
                source_sha256=sources,
                copied_exact=True,
            )
        )
    args.manifest.write_text(
        json.dumps(
            dict(
                records=records,
                source=str(Path(__file__).resolve()),
                source_sha256=sha(Path(__file__)),
            ),
            indent=2,
        )
        + "\n"
    )
    print(args.manifest)


if __name__ == "__main__":
    main()
