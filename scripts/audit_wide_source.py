"""Audit fixed initial weights, unchanged control updates and opening gradients."""

import argparse
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from cue_capacity import make_arm
from opening_prior_objective import opening_gradient
from physical_field_model import infer, load_field
from PIL import Image, ImageDraw
from plausible_study import exact
from scale_style_balance_study import update
from wide_source_study import (
    CONTROL,
    METHODS,
    MODELREF,
    arms_from_root,
    batch,
    check,
    load_data,
    replay_state,
)

from lineartgen.training import lineart_state

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
root = p.parse_args().root.resolve()
check(root)
assert jax.devices()[0].platform != "cpu"
data, arms = load_data(root), arms_from_root(root)
reference, _ = load_field(MODELREF / "snapshots/step-6000/opening/raw")
old = make_arm(arms["standard"]["config"], reference)
old["gradient"] = opening_gradient()
rows = []
for k in range(6):
    b = batch(data, k, "standard")
    candidate = batch(data, k, "expanded")
    exact(b[1:], candidate[1:])
    for method in METHODS:
        # Each later check uses an untouched initial clone, not the updated control.
        initial, _ = load_field(root / f"initials/{method}")
        parent, _ = load_field(root / "reference")
        mb = b if method == "standard" else candidate
        exact(infer(initial, mb[0]), infer(parent, mb[0]))
        parts, g = arms[method]["gradient"](arms[method]["model"], *mb)
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves((parts, g)))
        assert max(float(abs(np.asarray(x)).max()) for x in jax.tree.leaves(g)) > 0
        if method == "expanded":
            opening_value = float(parts["opening_consistency"])
            assert opening_value >= 0
    actual = update(arms["standard"], *b)
    expected = update(old, *b)
    rows.append(
        dict(
            depth=k + 1,
            initial_prediction_exact=True,
            gradients_finite_nonzero=True,
            opening_consistency=opening_value,
            control_metrics=replay_state(actual, expected),
            teacher_audit=data["last_audit"],
        )
    )
state = replay_state(
    lineart_state.payload({"standard": arms["standard"]}, 6)["arms"]["standard"],
    lineart_state.payload({"standard": old}, 6)["arms"]["standard"],
)
changed_checks = []
preview_rows = []
for k in range(100):
    standard = batch(data, k, "standard")
    expanded = batch(data, k, "expanded")
    if any(data["last_audit"]["changed_examples"]):
        exact(standard[1:], expanded[1:])
        changed_checks.append(dict(step=k, audit=data["last_audit"].copy()))
        item = data["last_audit"]["changed_examples"].index(True)
        assert not np.asarray(standard[1][2])[item]
        field = np.asarray(standard[1][0])[item, 16:-16, 16:-16].astype(np.float32)
        teacher = (1 - 2 * np.clip(0.5 - 4 * field, 0, 1).reshape(256, 4, 256, 4).mean((1, 3)))[
            ..., None
        ]
        preview_rows.append(
            (
                k,
                item,
                np.asarray(standard[0].guides[0][item]),
                np.asarray(expanded[0].guides[0][item]),
                teacher,
            )
        )
        if len(changed_checks) == 4:
            break
assert len(changed_checks) == 4
preview = root / "source-preview"
preview.mkdir(exist_ok=False)
canvas = Image.new("RGB", (768, 4 * 286), "white")
pen = ImageDraw.Draw(canvas)
bindings = {}
for row, (step, item, original, expanded, teacher) in enumerate(preview_rows):
    file = preview / f"step-{step}-item-{item}.npz"
    np.savez_compressed(file, original=original, expanded=expanded, teacher=teacher)
    bindings[file.name] = sha(file)
    for col, (label, value) in enumerate(
        zip(
            ("standard source", "expanded source", "same aligned teacher"),
            (original, expanded, teacher),
            strict=True,
        )
    ):
        pen.text((col * 256 + 3, row * 286 + 3), f"step{step} item{item} {label}", fill="black")
        px = np.rint(np.clip((value[..., 0] + 1) * 127.5, 0, 255)).astype(np.uint8)
        canvas.paste(Image.fromarray(px).convert("RGB"), (col * 256, row * 286 + 25))
canvas.save(preview / "native-source.png")
write(
    preview / "manifest.json",
    dict(
        files={"native-source.png": sha(preview / "native-source.png")},
        arrays=bindings,
        native_scale=True,
    ),
)
batch(data, 5999, "standard")
late = data["last_audit"].copy()
review = read(root / "teacher-preview/visual-review.json")
for f, h in review["files"].items():
    assert sha(root / "teacher-preview" / f) == h == sha(CONTROL / "teacher-preview" / f)
write(
    root / "preflight.json",
    dict(
        passed=True,
        changed_batch_checks=changed_checks,
        rows=rows,
        control_full_state_replay=state,
        final_batch_audit=late,
        teacher_preview_inherited=True,
        teacher_preview_sha256=sha(root / "teacher-preview/visual-review.json"),
    ),
)
print(dict(preflight=True, control_full_state_replay=state), flush=True)
