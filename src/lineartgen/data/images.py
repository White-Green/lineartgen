"""Demand-driven decoding for very large source PNGs; optional tiled TIFF preparation."""

import json
from pathlib import Path

import numpy as np
import pyvips


def image_size(path: Path) -> tuple[int, int]:
    image = pyvips.Image.new_from_file(str(path), access="sequential")
    return image.width, image.height


def read_crop(path: Path, x: int, y: int, size: int, *, alpha_mode="ignore"):
    access = "random" if path.suffix.lower() in (".tif", ".tiff") else "sequential"
    image = pyvips.Image.new_from_file(str(path), access=access)
    image = image.crop(x, y, size, size)
    if image.format != "uchar":
        raise ValueError(f"Expected 8-bit image data: {path} ({image.format})")
    pixels = np.frombuffer(image.write_to_memory(), np.uint8).reshape(size, size, image.bands)
    if image.bands in (1, 2):
        gray = pixels[..., 0].astype(np.float32) / 255
    elif image.bands in (3, 4):
        gray = (
            pixels[..., :3].astype(np.float32)
            @ np.array([0.2126, 0.7152, 0.0722], np.float32)
            / 255
        )
    else:
        raise ValueError(f"Unsupported channel count: {image.bands} ({path})")
    if alpha_mode == "white" and image.bands in (2, 4):
        alpha = pixels[..., -1].astype(np.float32) / 255
        gray = gray * alpha + 1 - alpha
    return gray[..., None]


def prepare_image(source: Path, destination: Path) -> None:
    """Convert once to lossless tiled TIFF for efficient random crop access.

    Write through a temporary file: interrupted preprocessing never produces a
    seemingly complete cache entry. Source pixels and alpha are preserved.
    """
    if destination.exists() and cache_matches(source, destination):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial.tif")
    image = pyvips.Image.new_from_file(str(source), access="sequential")
    image.tiffsave(
        str(temporary),
        tile=True,
        tile_width=min(512, (image.width + 15) // 16 * 16),
        tile_height=min(512, (image.height + 15) // 16 * 16),
        compression="deflate",
        bigtiff=True,
    )
    temporary.replace(destination)
    destination.with_suffix(".json").write_text(json.dumps(source_signature(source)) + "\n")


def source_signature(source):
    stat = source.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "format": 1}


def cache_matches(source, destination):
    manifest = destination.with_suffix(".json")
    return manifest.is_file() and json.loads(manifest.read_text()) == source_signature(source)
