import copy
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from compare_paper_guard_study import combine_panels, paired_config


def test_configuration_pairing_rejects_undeclared_changes():
    base = {"model": {"gradient": "hard"}, "loss": {"paper": 0}, "train": {"seed": 42}}
    current = copy.deepcopy(base)
    current["loss"]["paper"] = 0.3
    paired_config(current, base, {"loss": {"paper": 0.3}})
    assert base["loss"]["paper"] == 0
    current["train"]["seed"] = 43
    with pytest.raises(ValueError, match="declared"):
        paired_config(current, base, {"loss": {"paper": 0.3}})


def test_multi_root_panels_keep_pixels_and_check_all_inputs(tmp_path):
    size = 16

    def panel(path, values):
        pixels = np.full((size + 24, size * len(values)), 255, np.uint8)
        for i, value in enumerate(values):
            pixels[24:, i * size : (i + 1) * size] = value
        Image.fromarray(pixels).save(path)

    own, hard, ste, output = [tmp_path / f"{n}.png" for n in ("own", "hard", "ste", "output")]
    panel(own, [1, 2, 3, 11, 12])
    panel(hard, [1, 2, 3, 6, 7])
    panel(ste, [1, 2, 3, 8, 9])
    sources = [(own, ["a", "b"]), (hard, ["control", "unused"]), (ste, ["ste_control", "ste_ink"])]
    combine_panels(sources, output, ["control", "ste_ink", "a", "b"])
    with Image.open(output) as image:
        assert list(np.asarray(image)[24, ::size]) == [1, 2, 3, 6, 9, 11, 12]
    panel(ste, [1, 22, 3, 8, 9])
    with pytest.raises(AssertionError):
        combine_panels(sources, output, ["control", "ste_ink", "a", "b"])
