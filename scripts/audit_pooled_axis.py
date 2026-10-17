"""HI: preserve the HH initialization while activating a learned coarse pathway."""

import argparse
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from flax import nnx
from learned_coarse_axis_model import infer, load_field
from plausible_study import exact
from pooled_axis_study import GW, arms_from_root, batch, check, load_data

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
root = p.parse_args().root.resolve()
check(root)
assert jax.devices()[0].platform != "cpu"
data, arms = load_data(), arms_from_root(root)
base, _ = load_field(GW / "initials/choice")
model = arms["consistent"]["model"]
state = nnx.to_pure_dict(nnx.state(model, nnx.Param))
extra = state.pop("coarse_branch")
exact(nnx.to_pure_dict(nnx.state(base, nnx.Param)), state)
assert all(np.count_nonzero(np.asarray(v)) == 0 for v in jax.tree.leaves(extra["project"]))
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
    gc = float(np.linalg.norm(np.asarray(grad.coarse_branch.project.kernel)))
    assert gp > 0 and gr > 0 and gc > 0
    rows.append(
        dict(
            step=step,
            family=data["last_audit"]["family"],
            extension=data["coherent_extension"],
            initial_pixels_exact=True,
            pooled_gradient_norm=gp,
            residual_gradient_norm=gr,
            coarse_projection_gradient_norm=gc,
        )
    )
reuse = read(root / "teacher-reuse.json")
assert reuse["pages"] == 3 and reuse["new_pages_viewed"] == 0
for f, h in reuse["files"].items():
    assert sha(Path(f)) == h
    assert sha(root / "support-preview" / Path(f).name) == h
for step in (5994, 5999):
    batch(data, step)
write(
    root / "preflight.json",
    dict(
        passed=True,
        rows=rows,
        original_parameters_exact=True,
        new_projection_zero=True,
        reused_teacher_pages=3,
        new_teacher_pages=0,
        own36_exact_required=True,
        last_batches_checked=True,
    ),
)
print(dict(preflight=True, reused_teacher_pages=3), flush=True)
