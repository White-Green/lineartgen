"""Assemble three-condition S figures from verified, unscaled diagnostic pixels."""

import argparse
import json
from pathlib import Path

import numpy as np
from boundary_reuse import verify_reuse
from PIL import Image, ImageDraw
from summarize_training_methods import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    reference = verify_reuse(root, 10000)
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for mode in ("raw", "ema"):
        sources = [(reference, "control"), (root, "half"), (reference, "boundary")]
        evidence = []
        for folder, method in sources:
            path = folder / "evaluation" / f"widths-10000-{method}-{mode}/verification.json"
            audit = json.loads(path.read_text())
            assert audit["method"] == method and audit["mode"] == mode
            assert audit["metrics_sha256"] == sha(
                folder / "evaluation" / f"step-10000-{mode}/metrics.json"
            )
            assert audit["weights_sha256"] == sha(
                folder / "snapshots/step-10000" / method / mode / "weights.npz"
            )
            evidence.append((path, audit))
        for case in (256, 479, 480, 1376):
            panels, inputs = [], {}
            for path, audit in evidence:
                row = next(r for r in audit["records"] if r["case"] == case)
                image = Path(row["image"])
                assert sha(image) == row["sha256"] and row["recursion_levels"] == 3
                with Image.open(image) as opened:
                    pixels = np.asarray(opened).copy()
                assert pixels.shape == (536, 3072)
                panels.append(pixels[24:])
                inputs[str(image)] = sha(image)
                inputs[str(path)] = sha(path)
            for panel in panels[1:]:
                np.testing.assert_array_equal(panel[:, :1536], panels[0][:, :1536])
            pixels = np.concatenate(
                [panels[0][:, :1536], *(p[:, 1536:2048] for p in panels)], axis=1
            )
            canvas = Image.new("L", (3072, 536), 255)
            canvas.paste(Image.fromarray(pixels), (0, 24))
            draw = ImageDraw.Draw(canvas)
            labels = (
                "target (preserve mask)",
                "rough source",
                "noisy input",
                f"S lambda=0 {mode}",
                f"S lambda=0.2603685 {mode}",
                f"S lambda=0.520737 {mode}",
            )
            for index, label in enumerate(labels):
                draw.text((512 * index + 8, 5), label, fill=0)
            output = args.output / f"three-{mode}-case-{case:03d}.png"
            canvas.save(output)
            records.append(
                dict(
                    case=case,
                    mode=mode,
                    output=str(output.resolve()),
                    sha256=sha(output),
                    source_sha256=inputs,
                    inputs_exact=True,
                )
            )
    (args.output / "verification.json").write_text(
        json.dumps(dict(records=records, source_sha256=sha(Path(__file__))), indent=2) + "\n"
    )
    print(args.output)


if __name__ == "__main__":
    main()
