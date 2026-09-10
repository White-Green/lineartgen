"""Grain input pipeline: fixed shapes and index-derived randomness make resume deterministic.

Each logical optimizer batch samples one source scale. Images and crops are
independently sampled with replacement. The scale selects the recursion depth.
The validation split is by source filename, never by crop or resolution.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import grain
import numpy as np

from lineartgen.config import DataConfig

from .images import cache_matches, image_size, prepare_image, read_crop

SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass(frozen=True)
class Source:
    name: str
    variants: tuple[tuple[Path, int, int], ...]


@dataclass(frozen=True)
class Batch:
    images: np.ndarray
    scale_level: int


def discover(config: DataConfig):
    directories = [config.root / name for name in config.scale_dirs]
    files = [
        {p.name: p for p in directory.iterdir() if p.suffix.lower() in SUFFIXES}
        for directory in directories
    ]
    if not files[0]:
        raise ValueError(f"No images found in {directories[0]}")
    for directory, candidates in zip(directories, files, strict=True):
        if candidates.keys() != files[0].keys():
            raise ValueError(f"Source filenames do not match in {directory}")
    sources = []
    for name in sorted(files[0]):
        variants = []
        base_width = base_height = None
        base_level = config.resolved_scale_levels[0]
        for candidates, level in zip(files, config.resolved_scale_levels, strict=True):
            path = candidates[name]
            width, height = image_size(path)
            if base_width is None:
                base_width, base_height = width, height
            if (
                width * 2**base_level != base_width * 2**level
                or height * 2**base_level != base_height * 2**level
            ):
                raise ValueError(
                    f"Image dimensions do not match dataset scale level {level}: {path}"
                )
            if min(width, height) < config.crop_size:
                raise ValueError(f"Image smaller than crop_size: {path} ({width}x{height})")
            variants.append((path, width, height))
        sources.append(Source(name, tuple(variants)))
    return sources


def split_sources(sources, validation_count, seed):
    if validation_count >= len(sources):
        raise ValueError("validation_count must leave at least one training source")
    order = np.random.default_rng(seed).permutation(len(sources))
    return (
        [sources[i] for i in order[validation_count:]],
        [sources[i] for i in order[:validation_count]],
    )


def split_experiment(sources, config, seed):
    """Reserve final-test sources before training; changing model seeds cannot move them."""
    training, held_out = split_sources(
        sources,
        config.validation_count + config.final_test_count,
        seed if config.split_seed is None else config.split_seed,
    )
    return training, held_out[: config.validation_count], held_out[config.validation_count :]


def dataset_description(sources, training, validation, final_test=()):
    manifest = [
        [
            source.name,
            [
                (p.parent.name, w, h, p.stat().st_size, p.stat().st_mtime_ns)
                for p, w, h in source.variants
            ],
        ]
        for source in sources
    ]
    description = {
        "fingerprint": hashlib.sha256(json.dumps(manifest).encode()).hexdigest(),
        "train": [s.name for s in training],
        "validation": [s.name for s in validation],
        "sampling": (
            "uniform source with replacement; one scale per logical batch; "
            "indexed by optimizer step/item"
        ),
    }
    if final_test:
        description["final_test"] = [s.name for s in final_test]
    return description


@dataclass(frozen=True)
class LoadSample:
    sources: list[Source]
    config: DataConfig
    batch_size: int
    seed: int
    training: bool

    def __call__(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index), 0]))
        source = self.sources[rng.integers(len(self.sources))]
        scale_rng = np.random.default_rng([self.seed, int(index) // self.batch_size, 1])
        variant_index = int(scale_rng.integers(len(source.variants)))
        path, width, height = source.variants[variant_index]
        if self.config.prepared_root is not None:
            prepared = self.config.prepared_root / path.relative_to(self.config.root).with_suffix(
                path.suffix + ".tif"
            )
            if not prepared.is_file() or not cache_matches(path, prepared):
                raise FileNotFoundError(
                    f"Prepared image missing or stale: {prepared}; run lineartgen prepare"
                )
            path = prepared
        size = self.config.crop_size
        x, y = rng.integers(width - size + 1), rng.integers(height - size + 1)
        image = read_crop(path, int(x), int(y), size, alpha_mode=self.config.alpha_mode)
        if self.training and self.config.augment:
            image = np.rot90(image, int(rng.integers(4)))
            if rng.random() < 0.5:
                image = image[:, ::-1]
        return {
            "images": np.ascontiguousarray(
                np.clip((image * 2 - 1) * self.config.image_scale, -1, 1)
            ),
            "scale_level": np.int32(self.config.resolved_scale_levels[variant_index]),
        }


def batches(sources, config, *, batch_size, count, seed, start=0, training=True):
    dataset = grain.MapDataset.range(start * batch_size, (start + count) * batch_size)
    dataset = dataset.map(LoadSample(sources, config, batch_size, seed, training))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    stream = iter(
        dataset.to_iter_dataset(
            grain.ReadOptions(
                num_threads=max(1, config.workers),
                prefetch_buffer_size=max(config.workers, config.prefetch_batches),
            )
        )
    )
    try:
        for value in stream:
            yield Batch(value["images"], int(value["scale_level"][0]))
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            close()


def prepare_dataset(config: DataConfig, output: Path, limit: int | None = None):
    for index, source in enumerate(discover(config)):
        if limit is not None and index >= limit:
            break
        for path, _, _ in source.variants:
            destination = output / path.relative_to(config.root).with_suffix(path.suffix + ".tif")
            prepare_image(path, destination)
            yield destination
