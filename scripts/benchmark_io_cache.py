"""Compare lossless tiled TIFF with PNG on training sources, without changing training."""

import argparse
import fcntl
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyvips
from campaign_protocol import read, sha, write

from lineartgen.config import load_config
from lineartgen.data.images import cache_matches, prepare_image, read_crop
from lineartgen.data.pipeline import batches, discover, split_experiment


def benchmark(root, *, source_count=4, batch_count=36):
    root = root.resolve()
    assert read(root / "status.json")["stage"] == "stop_requested", "Pause training for isolation"
    config = load_config(root / "config.json")
    sources = discover(config.data)
    train, development, final = split_experiment(sources, config.data, config.train.seed)
    # Encoded size ranks cover different source complexity; selection uses no model predictions.
    ordered = sorted(train, key=lambda s: s.variants[-1][0].stat().st_size)
    indices = np.linspace(0, len(ordered) - 1, source_count).round().astype(int)
    selected = [ordered[i] for i in indices]
    assert len({s.name for s in selected}) == source_count
    assert not {s.name for s in selected} & {s.name for s in (*development, *final)}
    folder = root / "io-comparison"
    cache = folder / "tiles"
    folder.mkdir(exist_ok=True)
    conversions = []
    for source in selected:
        for original, width, height in source.variants:
            destination = cache / original.relative_to(config.data.root).with_suffix(
                original.suffix + ".tif"
            )
            metadata = folder / "conversion" / source.name / (original.parent.name + ".json")
            if metadata.exists():
                row = read(metadata)
                assert cache_matches(original, destination)
                assert sha(destination) == row["tiff_sha256"]
            else:
                start = time.perf_counter()
                prepare_image(original, destination)
                seconds = time.perf_counter() - start
                row = dict(
                    source=source.name,
                    scale_dir=original.parent.name,
                    width=width,
                    height=height,
                    seconds=seconds,
                    png_bytes=original.stat().st_size,
                    tiff_bytes=destination.stat().st_size,
                    png_sha256=sha(original),
                    tiff_sha256=sha(destination),
                )
                write(metadata, row)
            conversions.append(row)
            print(
                f"Prepared {source.name}/{original.parent.name}: {row['seconds']:.2f}s", flush=True
            )
    random = np.random.default_rng(20260919)
    readers = []

    def load(task):
        path, x, y = task
        gray = read_crop(path, x, y, config.data.crop_size, alpha_mode=config.data.alpha_mode)
        return np.ascontiguousarray(np.clip((gray * 2 - 1) * config.data.image_scale, -1, 1))

    with ThreadPoolExecutor(max_workers=config.data.workers) as pool:
        for scale in range(len(config.data.scale_dirs)):
            for repeat in range(3):
                jobs = []
                for i in range(config.train.effective_batch_size):
                    path, width, height = selected[i % len(selected)].variants[scale]
                    size = config.data.crop_size
                    x, y = (
                        int(random.integers(width - size + 1)),
                        int(random.integers(height - size + 1)),
                    )
                    tile = cache / path.relative_to(config.data.root).with_suffix(
                        path.suffix + ".tif"
                    )
                    jobs.append((path, tile, x, y))
                results, times = {}, {}
                for kind in ("png", "tiff") if repeat % 2 == 0 else ("tiff", "png"):
                    tasks = [(p if kind == "png" else t, x, y) for p, t, x, y in jobs]
                    start = time.perf_counter()
                    results[kind] = list(pool.map(load, tasks))
                    times[kind] = time.perf_counter() - start
                for png, tiff in zip(results["png"], results["tiff"], strict=True):
                    np.testing.assert_array_equal(png, tiff)
                row = dict(
                    scale=scale,
                    repeat=repeat,
                    batch_size=len(jobs),
                    seconds=times,
                    pixels_exact=True,
                )
                readers.append(row)
                print(json.dumps(row), flush=True)
    pipelines = []
    configurations = {
        "png": config.data,
        "tiff": config.data.model_copy(update={"prepared_root": cache}),
    }
    for delay in (0.0, 1.15):
        expected = None
        for kind in ("png", "tiff") if delay == 0 else ("tiff", "png"):
            iterator = batches(
                selected,
                configurations[kind],
                batch_size=config.train.effective_batch_size,
                count=batch_count,
                seed=config.train.seed,
                training=True,
            )
            durations, digests, scales = [], [], []
            start_all = time.perf_counter()
            try:
                for _ in range(batch_count):
                    start = time.perf_counter()
                    batch = next(iterator)
                    durations.append(time.perf_counter() - start)
                    digests.append(hashlib.sha256(batch.images.tobytes()).hexdigest())
                    scales.append(batch.scale_level)
                    if delay:
                        time.sleep(delay)
            finally:
                iterator.close()
            total = time.perf_counter() - start_all
            signature = list(zip(scales, digests, strict=True))
            if expected is None:
                expected = signature
            else:
                assert signature == expected, "Indexed Grain pixels/augmentations differ"
            row = dict(
                format=kind,
                simulated_consumer_seconds=delay,
                batch_count=batch_count,
                input_wait_seconds=durations,
                total_seconds=total,
                steady_mean_wait_seconds=float(np.mean(durations[6:])),
                steady_median_wait_seconds=float(np.median(durations[6:])),
                scale_levels=scales,
                batch_sha256=digests,
            )
            pipelines.append(row)
            print(
                f"Grain {kind}, consumer {delay}s: {total:.2f}s; "
                f"wait mean {row['steady_mean_wait_seconds']:.5f}s",
                flush=True,
            )
    result = dict(
        sources=[s.name for s in selected],
        selection="Four training-source ranks by largest PNG byte size; no held-out pixels read.",
        conversions=conversions,
        readers=readers,
        pipelines=pipelines,
        source_sha256=sha(__file__),
        config_sha256=sha(root / "config.json"),
        vips_version=".".join(str(pyvips.version(i)) for i in range(3)),
        threads=config.data.workers,
        prefetch=config.data.prefetch_batches,
        pixels_exact=True,
        grain_batches_exact=True,
        conversion_seconds=sum(r["seconds"] for r in conversions),
        png_bytes=sum(r["png_bytes"] for r in conversions),
        tiff_bytes=sum(r["tiff_bytes"] for r in conversions),
        limitations="Small, warm-cache source sample. OS caches are not dropped. "
        "Reader rounds alternate format order; distinct crops prevent identical-result reuse. "
        "Grain consumer delay is a simulation, not a GPU update. Training remains unchanged.",
    )
    write(folder / "benchmark.json", result)
    return folder / "benchmark.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    assert (args.root / "stop").exists(), "Use the campaign's checkpoint-safe stop first"
    with (args.root / "controller.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(benchmark(args.root))


if __name__ == "__main__":
    main()
