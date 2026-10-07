import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from compare_output_gradient_study import align_rows, combine_panel


def test_external_baseline_requires_identical_case_conditions():
    baseline = dict(
        case=3,
        method="control",
        domain="geometry",
        kind="curve",
        source=None,
        suite="synthetic",
        family="strokes",
        severity="strong",
        strength=0.5,
        partial=False,
        denoise_steps=1,
        halo_opacity_excess=0.2,
    )
    current = baseline | dict(method="learned_features", halo_opacity_excess=0.1)
    align_rows([current], [baseline], ["learned_features"])
    for changed in (dict(strength=0.1), dict(partial=True), dict(case=4)):
        with pytest.raises(ValueError):
            align_rows([current | changed], [baseline], ["learned_features"])
    with pytest.raises(ValueError):
        align_rows([current, current], [baseline], ["learned_features"])
    with pytest.raises(ValueError):
        align_rows([current], [baseline, baseline], ["learned_features"])


def test_comparison_panels_preserve_pixels_and_reject_different_inputs(tmp_path):
    size = 16

    def panel(path, values):
        pixels = np.full((size + 24, size * len(values)), 255, np.uint8)
        for i, value in enumerate(values):
            pixels[24:, i * size : (i + 1) * size] = value
        Image.fromarray(pixels).save(path)

    own, parent, output = [tmp_path / f"{name}.png" for name in ("own", "parent", "output")]
    panel(own, [1, 2, 3, 9])
    panel(parent, [1, 2, 3, 6, 7, 8])
    combine_panel(own, parent, output, ["features"], ["control", "regions", "ink_guard"])
    with Image.open(output) as image:
        assert list(np.asarray(image)[24, ::size]) == [1, 2, 3, 6, 8, 9]
    panel(parent, [1, 2, 4, 6, 7, 8])
    with pytest.raises(AssertionError):
        combine_panel(own, parent, output, ["features"], ["control", "regions", "ink_guard"])
