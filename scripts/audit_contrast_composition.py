"""Audit compound field targets and untouched examples at all recursive depths."""

import argparse
from pathlib import Path

import jax
import merge_geometry_study as previous
import numpy as np
from campaign_protocol import read, sha, write
from continuous_width_data import width_choice, width_source
from contrast_composition_data import overlay_source
from contrast_composition_study import CONTROL, SEED, batch, components, load_data, select

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
root = p.parse_args().root.resolve()
data = load_data(root)
changed = 0
for step in range(6):
    selection = select(step)
    src, clean, fields = components(data, selection, "mixed", step)
    base = previous.components(data, selection, "merge", step)
    for item, (i, overlay) in enumerate(
        zip(selection["indices"], selection["overlays"], strict=True)
    ):
        if overlay["enabled"]:
            changed += 1
            j = overlay["index"]
            secondary = width_source(
                data, data["records"][j], j, overlay["width"], overlay["augmentation"]
            )
            np.testing.assert_array_equal(src[item], overlay_source(base[0][item], secondary))
            np.testing.assert_array_equal(
                fields[item], np.minimum(data["fields"][i], data["fields"][j])
            )
            np.testing.assert_array_equal(
                clean[item], np.minimum(data["variants"][i, 13], data["variants"][j, 13])
            )
        else:
            for x, y in zip((src, clean, fields), base, strict=True):
                np.testing.assert_array_equal(x[item], y[item])
        assert width_choice(SEED, step, item, i) == width_choice(previous.SEED, step, item, i)
    a = batch(data, step, "mixed")
    b = previous.batch(data, step, "merge")
    np.testing.assert_array_equal(a[2], b[2])
    np.testing.assert_array_equal(a[1][2], b[1][2])
    jax.tree.map(np.testing.assert_array_equal, a[0].noise_levels, b[0].noise_levels)
    for target, guide, noise, base_guide, base_noise, mask in zip(
        a[0].targets,
        a[0].guides,
        a[0].noisy,
        b[0].guides,
        b[0].noisy,
        a[0].noise_levels,
        strict=True,
    ):
        protected = np.asarray(mask) == 0
        np.testing.assert_array_equal(np.asarray(target)[protected], np.asarray(guide)[protected])
        np.testing.assert_allclose(noise - guide, base_noise - base_guide, atol=1.2e-7, rtol=0)
    unchanged = [i for i, x in enumerate(selection["overlays"]) if not x["enabled"]]
    if unchanged:
        jax.tree.map(
            lambda x, y, ii=unchanged: np.testing.assert_array_equal(
                np.asarray(x)[ii], np.asarray(y)[ii]
            ),
            a,
            b,
        )
assert changed > 0
inherited = read(CONTROL / "test-results.json")
assert inherited["passed"]
for name in [
    "merge_field_objective",
    "merge_field_data",
    "group_field_model",
    "continuous_width_data",
    "d4_geometry_data",
]:
    f = Path("scripts") / (name + ".py")
    assert sha(f) == sha(CONTROL.parent.parent / f)
write(
    root / "preflight.json",
    dict(
        passed=True,
        six_depth_inputs_and_targets_verified=True,
        composed_examples_checked=changed,
        inherited_tests=inherited,
        inherited_preflight_sha256=sha(CONTROL / "preflight.json"),
    ),
)
