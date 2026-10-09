import numpy as np
import pytest
from PIL import Image

from lineartgen.config import DataConfig
from lineartgen.data.pipeline import LoadSample, batches, discover


def test_native_teachers_keep_virtual_depth_and_resume(tmp_path):
    # Deliberately shuffled mapping: native does not mean the first directory.
    dirs, levels = ("eight", "native", "two", "four", "sixteen", "thirtytwo"), (3, 0, 1, 2, 4, 5)
    for folder, level in zip(dirs, levels, strict=True):
        (tmp_path / folder).mkdir()
        Image.new("L", (16 * 2**level, 16 * 2**level), 20 + 30 * level).save(
            tmp_path / folder / "sample.png"
        )
    default = DataConfig(
        root=tmp_path,
        scale_dirs=dirs,
        scale_levels=levels,
        crop_size=16,
        workers=1,
        augment=False,
        image_scale=1,
    )
    native = default.model_copy(update={"native_training_targets": True})
    sources = discover(default)
    ordinary = list(batches(sources, default, batch_size=2, count=60, seed=42))
    changed = list(batches(sources, native, batch_size=2, count=60, seed=42))
    assert {b.scale_level for b in changed} == set(range(6))
    for old, new in zip(ordinary, changed, strict=True):
        assert old.scale_level == new.scale_level
        np.testing.assert_allclose(old.images, (20 + 30 * old.scale_level) / 255 * 2 - 1, atol=1e-7)
        np.testing.assert_allclose(new.images, 20 / 255 * 2 - 1, atol=1e-7)
    resumed = list(batches(sources, native, batch_size=2, count=4, start=5, seed=42))
    for original, restored in zip(changed[5:9], resumed, strict=True):
        np.testing.assert_array_equal(original.images, restored.images)
        assert original.scale_level == restored.scale_level
    # Evaluation keeps its original resolution selection.
    for i in range(20):
        old = LoadSample(sources, default, 2, 42, False)(i)
        new = LoadSample(sources, native, 2, 42, False)(i)
        np.testing.assert_array_equal(old["images"], new["images"])
        assert old["scale_level"] == new["scale_level"]


def test_native_requires_native_scale():
    with pytest.raises(ValueError, match="level-0"):
        DataConfig(scale_dirs=("two",), scale_levels=(1,), native_training_targets=True)
