"""Check unchanged inputs, teacher geometry, finite gradients and matched control replay."""

import argparse
from pathlib import Path

import jax
import numpy as np
from aligned_geometry_data import component_field
from aligned_geometry_study import CONTROL, arms_from_root, batch, check, load_data, replay_state
from campaign_protocol import sha, write
from physical_field_model import infer, load_field
from PIL import Image, ImageDraw
from plausible_study import exact
from scale_style_balance_study import update
from subpixel_field_data import central_field
from widened_geometry_study import arms_from_root as old_arms
from widened_geometry_study import batch as old_batch

from lineartgen.training import checkpoints, lineart_state

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
root = p.parse_args().root.resolve()
check(root)
assert jax.devices()[0].platform != "cpu"
data, new, old = load_data(root), arms_from_root(root), old_arms(CONTROL)
reference, _ = load_field(root / "reference")
rows, previews = [], []
dest = root / "teacher-preview"
dest.mkdir(exist_ok=False)


def oracle(targets):
    field = np.where(np.asarray(targets[2])[:, None, None], targets[1], targets[0])
    alpha = np.clip(0.5 - 4 * np.asarray(central_field(field), np.float32), 0, 1)
    return (1 - 2 * alpha.reshape(4, 256, 4, 256, 4).mean((2, 4)))[..., None]


for k in range(6):
    a, b = batch(data, k, "aligned"), old_batch(data, k, "control")
    exact(a[0], b[0])
    exact(a[2], b[2])
    exact(a[1][2:], b[1][2:])
    small = new["aligned"]["model"]
    exact(infer(small, a[0]), infer(reference, a[0]))
    parts, g = new["aligned"]["gradient"](small, *a)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves((parts, g)))
    assert max(float(abs(np.asarray(x)).max()) for x in jax.tree.leaves(g)) > 0
    previous = update(old["control"], *b)
    rows.append(
        dict(
            depth=k + 1,
            initial_prediction_exact=True,
            prepared_inputs_exact=True,
            mask_and_directions_exact=True,
            gradient_finite=True,
            loss=float(parts["loss"]),
            teacher_audit=data["last_audit"],
            inherited_metrics=replay_state(previous, data["control_rows"][k]["metrics"]["control"]),
        )
    )
    if k in (0, 5):
        canvas = Image.new("RGB", (768, 36 + 4 * 280), "white")
        pen = ImageDraw.Draw(canvas)
        values = [np.asarray(a[0].guides[0]), oracle(b[1]), oracle(a[1])]
        for col, (label, images) in enumerate(
            zip(("source", "canonical teacher", "aligned teacher"), values, strict=True)
        ):
            pen.text((col * 256 + 6, 6), label, fill="black")
            for item, image in enumerate(images):
                image = np.where(np.asarray(a[0].noise_levels[0])[item] > 0, image, values[0][item])
                gray = np.uint8(np.rint(np.clip((image[..., 0] + 1) * 127.5, 0, 255)))
                canvas.paste(Image.fromarray(gray).convert("RGB"), (col * 256, 60 + item * 280))
                pen.text((col * 256 + 6, 38 + item * 280), f"batch{k} item{item}", fill="black")
        file = dest / f"batch-{k}.png"
        canvas.save(file)
        previews.append(dict(file=file.name, sha256=sha(file)))
for i in (0, 288, 432, 448):
    zero = dict(amplitude=0, phase_x=0, phase_y=0, period=96)
    np.testing.assert_array_equal(component_field(data, i, zero), data["fields"][i])
    if i in data["merge_index"]:
        np.testing.assert_array_equal(
            component_field(data, i, zero, merged=True),
            data["merged_fields"][data["merge_index"][i]],
        )
    if i in (432, 448):
        np.testing.assert_array_equal(
            component_field(data, i, zero | dict(amplitude=1.5)), data["fields"][i]
        )
restored = old_arms(CONTROL)
with checkpoints.manager(CONTROL / "training/checkpoints", keep=5) as saved:
    assert lineart_state.restore(saved, restored, 6) == 6
state = replay_state(
    lineart_state.payload(old, 6)["arms"]["control"],
    lineart_state.payload(restored, 6)["arms"]["control"],
)
for name in (
    "forced_context_objective",
    "quartic_field_objective",
    "context_dropout_objective",
    "joint_geometry_study",
    "physical_field_model",
    "group_field_model",
    "subpixel_field_model",
    "support_merge_data",
    "scale_style_balance_study",
):
    file = Path("scripts") / (name + ".py")
    assert sha(file) == sha(CONTROL.parent.parent / file)
write(
    dest / "manifest.json",
    dict(records=previews, native_size=256, thresholding=False, contrast_rescaling=False),
)
write(
    root / "preflight.json",
    dict(
        passed=True,
        rows=rows,
        zero_warp_cached_exact=True,
        fill_blank_unchanged=True,
        control_full_state_replay=state,
    ),
)
print(dict(preflight=True, control_full_state_replay=state), flush=True)
