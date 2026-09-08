import numpy as np
import pytest
from PIL import Image

from lineartgen.config import DataConfig, ExperimentConfig
from lineartgen.data.images import prepare_image, read_crop
from lineartgen.data.pipeline import batches, discover, split_sources


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "dataset"
    for scale, size in (("images", 32), ("images_2x", 64)):
        (root / scale).mkdir(parents=True)
        for index in range(8):
            random = np.random.default_rng(index)
            pixels = random.integers(0, 256, (size, size, 4), np.uint8)
            Image.fromarray(pixels).save(root / scale / f"{index:04}.png")
    return root


def test_split_and_resume_input_stream(dataset):
    config = DataConfig(
        root=dataset,
        scale_dirs=("images", "images_2x"),
        crop_size=16,
        validation_count=2,
        workers=2,
    )
    train, validation = split_sources(discover(config), 2, 42)
    assert not {s.name for s in train} & {s.name for s in validation}
    full = list(batches(train, config, batch_size=2, count=4, seed=42))
    resumed = list(batches(train, config, batch_size=2, count=2, start=2, seed=42))
    for original, restored in zip(full[2:], resumed, strict=True):
        np.testing.assert_array_equal(original.images, restored.images)
        assert original.scale_level == restored.scale_level


def test_tiled_preparation_preserves_alpha_and_pixels(dataset, tmp_path):
    source = dataset / "images" / "0000.png"
    destination = tmp_path / "cache" / "0000.png.tif"
    prepare_image(source, destination)
    for alpha_mode in ("ignore", "white"):
        np.testing.assert_array_equal(
            read_crop(source, 3, 7, 16, alpha_mode=alpha_mode),
            read_crop(destination, 3, 7, 16, alpha_mode=alpha_mode),
        )


def test_scale_subset_keeps_original_level(dataset):
    config = DataConfig(root=dataset, scale_dirs=("images_2x",), scale_levels=(1,), crop_size=16)
    stream = batches(discover(config), config, batch_size=4, count=2, seed=42)
    assert [batch.scale_level for batch in stream] == [1, 1]


def test_mismatched_scale_dimensions_are_rejected(dataset):
    Image.new("L", (63, 64)).save(dataset / "images_2x" / "0000.png")
    config = DataConfig(root=dataset, scale_dirs=("images", "images_2x"), crop_size=16)
    with pytest.raises(ValueError, match="dimensions do not match dataset scale"):
        discover(config)


@pytest.mark.parametrize(
    "settings",
    [
        {"data": {"crop_size": 48}},
        {"data": {"scale_dirs": []}},
        {"data": {"scale_dirs": ["images"], "scale_levels": []}},
        {"data": {"scale_dirs": ["images"], "scale_levels": [-1]}},
        {"data": {"scale_dirs": ["images", "images_2x"], "scale_levels": [0, 0]}},
        {"diffusion": {"original_steps": 0}},
        {"matching": {"pool_size": 8}},
        {"model": {"kernel_size": 2}},
        {"diffusion": {"pattern_weights": [0, 0, 0, 0]}},
        {"diffusion": {"pattern_weights": [float("nan"), 1, 1, 1]}},
        {"train": {"learning_rate": float("inf")}},
        {"typo": 1},
    ],
)
def test_invalid_configuration(settings):
    with pytest.raises(ValueError):
        ExperimentConfig.model_validate(settings)
