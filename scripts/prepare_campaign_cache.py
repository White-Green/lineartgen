"""Prepare and verify lossless train/development tiles; never decode reserved final sources."""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from campaign_protocol import read, sha, write

from lineartgen.config import load_config
from lineartgen.data.images import cache_matches, prepare_image, read_crop
from lineartgen.data.pipeline import dataset_description, discover, split_experiment


def prepare(root, destination, *, workers=4, reserve_gib=100):
    root, destination = root.resolve(), destination.resolve()
    assert read(root / "status.json")["stage"] == "stop_requested"
    config = load_config(root / "config.json")
    sources = discover(config.data)
    train, development, final = split_experiment(sources, config.data, config.train.seed)
    description = dataset_description(sources, train, development, final)
    assert description == read(root / "arms/seed42_post_pool_d1/training/model/dataset.json")
    allowed, forbidden = {s.name for s in (*train, *development)}, {s.name for s in final}
    assert len(allowed) == 955 and not allowed & forbidden
    destination.mkdir(parents=True, exist_ok=True)
    write(
        destination / "preparation-source.json",
        dict(
            source_sha256=sha(__file__),
            config_sha256=sha(root / "config.json"),
            dataset=description,
            workers=workers,
            reserve_gib=reserve_gib,
            selected_sources=sorted(allowed),
            excluded_final_sources=sorted(forbidden),
        ),
    )
    started = time.perf_counter()
    jobs = [
        (s, scale, variant)
        for s in (*train, *development)
        for scale, variant in enumerate(s.variants)
    ]

    def convert(job):
        source, scale, (path, width, height) = job
        assert source.name in allowed and source.name not in forbidden
        relative = path.relative_to(config.data.root)
        target = destination / relative.with_suffix(path.suffix + ".tif")
        audit = destination / "verification" / relative.with_suffix(path.suffix + ".json")
        if audit.exists() and target.exists() and cache_matches(path, target):
            previous = read(audit)
            assert sha(target) == previous["tiff_sha256"]
            assert sha(path) == previous["png_sha256"]
            return previous
        if shutil.disk_usage(destination).free < reserve_gib * 2**30:
            raise RuntimeError(f"Less than {reserve_gib} GiB free; preparation stopped safely")
        # Reuse the measured sample without duplicating its bytes on the same filesystem.
        sample = root / "io-comparison/tiles" / relative.with_suffix(path.suffix + ".tif")
        if not target.exists() and sample.exists() and cache_matches(path, sample):
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(sample, target)
            shutil.copyfile(sample.with_suffix(".json"), target.with_suffix(".json"))
        start = time.perf_counter()
        prepare_image(path, target)
        conversion_seconds = time.perf_counter() - start
        size = config.data.crop_size
        coordinates = [
            (0, 0),
            ((width - size) // 2, (height - size) // 2),
            (width - size, height - size),
        ]
        crops = []
        for x, y in coordinates:
            png = read_crop(path, x, y, size, alpha_mode=config.data.alpha_mode)
            tiff = read_crop(target, x, y, size, alpha_mode=config.data.alpha_mode)
            np.testing.assert_array_equal(png, tiff)
            for value in (png, tiff):
                assert value.dtype == np.float32 and np.isfinite(value).all()
            crops.append(dict(x=x, y=y, sha256=hashlib.sha256(png.tobytes()).hexdigest()))
        row = dict(
            source=source.name,
            scale=scale,
            source_path=str(path),
            destination=str(target),
            png_sha256=sha(path),
            tiff_sha256=sha(target),
            png_bytes=path.stat().st_size,
            tiff_bytes=target.stat().st_size,
            conversion_seconds=conversion_seconds,
            total_seconds=time.perf_counter() - start,
            crop_checks=crops,
            pixels_exact=True,
        )
        write(audit, row)
        return row

    records = []
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(convert, job) for job in jobs]
            try:
                for future in as_completed(futures):
                    row = future.result()
                    records.append(row)
                    if len(records) % 25 == 0 or len(records) == len(jobs):
                        status = dict(
                            stage="preparing",
                            completed=len(records),
                            expected=len(jobs),
                            elapsed_seconds=time.perf_counter() - started,
                            cache_bytes=sum(r["tiff_bytes"] for r in records),
                            free_bytes=shutil.disk_usage(destination).free,
                        )
                        write(destination / "preparation-status.json", status)
                        print(json.dumps(status), flush=True)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        assert len(records) == 5730
        expected = {(s.name, i) for s in (*train, *development) for i in range(6)}
        assert {(r["source"], r["scale"]) for r in records} == expected
        assert not any(
            (destination / d / name).with_suffix(".png.tif").exists()
            for d in config.data.scale_dirs
            for name in forbidden
        )
        result = dict(
            stage="complete",
            sources=len(allowed),
            files=len(records),
            elapsed_seconds=time.perf_counter() - started,
            png_bytes=sum(r["png_bytes"] for r in records),
            cache_bytes=sum(r["tiff_bytes"] for r in records),
            verified_crops=sum(len(r["crop_checks"]) for r in records),
            final_pixels_untouched=True,
            records=sorted(records, key=lambda r: (r["source"], r["scale"])),
            source_sha256=sha(__file__),
        )
        write(destination / "preparation.json", result)
        write(
            destination / "preparation-status.json",
            {k: v for k, v in result.items() if k != "records"},
        )
    except BaseException as error:
        write(
            destination / "preparation-status.json",
            dict(
                stage="failed",
                completed=len(records),
                expected=len(jobs),
                error=str(error),
                elapsed_seconds=time.perf_counter() - started,
            ),
        )
        raise
    return destination / "preparation.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reserve-gib", type=int, default=100)
    args = parser.parse_args()
    assert (args.root / "stop").exists()
    with (args.root / "controller.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(prepare(args.root, args.output, workers=args.workers, reserve_gib=args.reserve_gib))


if __name__ == "__main__":
    main()
