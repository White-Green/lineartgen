"""Measure deployed CPU inference, including opaque rendering, on larger canvases."""

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
from lineartgen_runtime.graph import GraphLineartModel, render
from PIL import Image, ImageDraw


def benchmark(assets, size, depth, output, threads=1):
    before = time.monotonic()
    model = GraphLineartModel(assets / "graph-model.json", threads, provider="cpu")
    loading = time.monotonic() - before
    drawing = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(drawing)
    for offset in range(80, size - 50, 160):
        draw.line([(40, offset), (size - 40, offset + 30)], fill=150, width=9)
        draw.line([(offset, 40), (offset + 30, size - 40)], fill=100, width=7)
    alpha = np.asarray(drawing)
    source = 1 - 2 * alpha.astype(np.float32)[None, ..., None] / 255
    noise = np.full_like(source, 0.5)
    before = time.monotonic()
    heads = model.heads(source, noise, depth)
    inference = time.monotonic() - before
    before = time.monotonic()
    result = render(heads[0][0])
    rendering = time.monotonic() - before
    assert np.isfinite(result).all() and np.any(result < 0)
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint((result[..., 0] + 1) * 127.5).astype(np.uint8)).save(
        output / f"{size}-d{depth}.png"
    )
    report = dict(
        size=size,
        depth=depth,
        provider=model.device,
        threads=threads,
        model_training_steps=model.metadata["training_steps"],
        loading_seconds=loading,
        heads_seconds=inference,
        rendering_seconds=rendering,
        total_seconds=inference + rendering,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        output_ink_pixels=int((result < 0).sum()),
        note="CPU timing; no DirectML performance claim. No BGRA padding in this benchmark.",
    )
    (output / f"{size}-d{depth}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    benchmark(**vars(parser.parse_args()))
