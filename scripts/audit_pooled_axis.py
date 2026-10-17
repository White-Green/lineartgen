"""HD: unchanged initialization, all families/depths, and new partial-axis teachers."""

import argparse
from pathlib import Path

import jax
import numpy as np
from ambiguous_shape_data import selected
from campaign_protocol import sha, write
from consistent_axis_fit import page
from consistent_axis_objective import preferred_original
from flax import nnx
from plausible_study import exact
from pooled_axis_model import infer, load_field
from pooled_axis_study import GW, arms_from_root, batch, check, load_data
from subpixel_field_data import candidate_fields
from subpixel_field_model import coverage

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
root = p.parse_args().root.resolve()
check(root)
assert jax.devices()[0].platform != "cpu"
data, arms = load_data(), arms_from_root(root)
base, _ = load_field(GW / "initials/choice")
model = arms["consistent"]["model"]
exact(nnx.state(base, nnx.Param), nnx.state(model, nnx.Param))
rows = []
for step in (0, 1, 2, 3, 4, 5, 18, 19, 20, 21, 22, 23, 30, 31, 32, 33, 34, 35):
    b = batch(data, step)
    prepared, c, d = b
    np.testing.assert_array_equal(infer(base, prepared), infer(model, prepared))
    parts, grad = arms["consistent"]["gradient"](model, *b)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves((parts, grad)))
    assert float(parts["axis_eligible_fraction"]) == float(np.asarray(c[6]).mean())
    gp = float(np.linalg.norm(np.asarray(grad.pooled_stem.kernel)))
    gr = float(np.linalg.norm(np.asarray(grad.head.residual.kernel)))
    assert gp > 0 and gr > 0
    rows.append(
        dict(
            step=step,
            family=data["last_audit"]["family"],
            extension=data["support_extension"],
            initial_pixels_exact=True,
            pooled_gradient_norm=gp,
            residual_gradient_norm=gr,
        )
    )

out = root / "support-preview"
out.mkdir(exist_ok=False)
examples, previews, labels, bindings = [], [], [], {}
for step in range(6000):
    if not selected(step):
        continue
    prepared, c, d = batch(data, step)
    flipped = np.asarray(data["support_extension"]["promoted"]) & ~np.asarray(preferred_original(d))
    if not flipped.any():
        continue
    original = np.asarray(coverage(candidate_fields(c[0], d, 1)[0][:, 4]))
    alternate = np.asarray(coverage(candidate_fields(c[4], c[5], 1)[0][:, 4]))
    for item in np.flatnonzero(flipped):
        source = np.asarray(prepared.guides[0])[item]
        strength = np.asarray(prepared.noise_levels[0])[item]
        assert np.any(strength == 0)
        old = np.where(strength > 0, original[item], source)
        new = np.where(strength > 0, alternate[item], source)
        f = out / f"example-{len(examples)}.npz"
        np.savez_compressed(f, source=source, strength=strength, original=old, supported=new)
        bindings[str(f)] = sha(f)
        examples.append(
            dict(step=step, item=int(item), recipe=data["last_audit"]["ambiguous_records"][item])
        )
        previews.append([source, old, new, 1 - 2 * (strength > 0).astype(np.float32)])
        labels.append(
            [
                f"step{step} item{item} source",
                "old original only",
                "supported preferred",
                "active mask",
            ]
        )
        if len(examples) == 8:
            break
    if len(examples) == 8:
        break
assert len(examples) == 8
records = []
for start in (0, 4):
    f = out / f"native-{start}.png"
    page(f, previews[start : start + 4], labels[start : start + 4])
    records.append(dict(file=f.name, sha256=sha(f)))
write(
    out / "manifest.json",
    dict(records=records, native_scale=True, examples=examples, bindings=bindings),
)
for step in (5994, 5999):
    batch(data, step)
write(
    root / "preflight.json",
    dict(
        passed=True,
        rows=rows,
        initial_parameters_exact=True,
        support_preview_examples=8,
        own36_exact_required=True,
        last_batches_checked=True,
    ),
)
print(dict(preflight=True, examples=8), flush=True)
