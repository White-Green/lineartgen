"""Check matched inputs, conditional geometry, and a real compiled model update."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import width_coverage_study as previous
from campaign_protocol import read, write
from flax import nnx
from merge_geometry_study import CONTROL, arms_from_root, batch, load_data, select
from plausible_study import digest, exact
from regional_field_objective import regional_gradient
from scale_style_balance_study import update
from subpixel_field_model import load_field, save_field


def inputs(root):
    data = load_data(root)
    chosen = {}
    for step in range(600):
        s = select(step)
        if s["depth"] in chosen or not any(i in data["merge_index"] for i in s["indices"]):
            continue
        p, bases, direction = batch(data, step, "merge")
        old = previous.batch(data, step, "width")
        jax.tree.map(np.testing.assert_array_equal, (p, bases[0], direction), old)
        for item, allowed in enumerate(np.asarray(bases[2])):
            if not allowed:
                np.testing.assert_array_equal(bases[0][item], bases[1][item])
        if np.asarray(bases[2]).any():
            chosen[s["depth"]] = dict(step=step, eligible=np.asarray(bases[2]).tolist())
        if len(chosen) == 6:
            break
    assert len(chosen) == 6
    write(
        root / "preflight-inputs.json",
        dict(
            passed=True,
            six_depth_inputs_exact=True,
            choices=chosen,
            cached_candidates=len(data["merge_index"]),
            control=str(CONTROL.resolve()),
        ),
    )


def gpu(root):
    assert jax.devices()[0].platform != "cpu"
    data = load_data(root)
    step = read(root / "preflight-inputs.json")["choices"]["3"]["step"]
    p, bases, direction = batch(data, step, "merge")
    arm = arms_from_root(root)["merge"]
    oldparts, oldgrads = regional_gradient()(arm["model"], p, bases[0], direction)
    closed = (bases[0], bases[1], jnp.zeros_like(bases[2]))
    sameparts, samegrads = arm["gradient"](arm["model"], p, closed, direction)
    maximum = 0.0
    for a, b in zip(jax.tree.leaves(oldgrads), jax.tree.leaves(samegrads), strict=True):
        np.testing.assert_allclose(a, b, atol=5e-6, rtol=1e-4)
        maximum = max(maximum, float(np.max(abs(np.asarray(a) - np.asarray(b)))))
    for k in oldparts:
        np.testing.assert_allclose(oldparts[k], sameparts[k], atol=5e-6, rtol=1e-4)
    before = digest(nnx.state(arm["model"], nnx.Param))
    metrics = update(arm, p, bases, direction)
    after = digest(nnx.state(arm["model"], nnx.Param))
    assert before != after
    destination = root / "preflight-export"
    save_field(
        arm["model"], arm["config"], destination, provenance="preflight only; discarded update"
    )
    restored, cfg = load_field(destination)
    assert cfg == arm["config"]
    exact(nnx.state(restored, nnx.Param), nnx.state(arm["model"], nnx.Param))
    write(
        root / "preflight.json",
        dict(
            passed=True,
            input_audit=read(root / "preflight-inputs.json"),
            gpu=jax.devices()[0].device_kind,
            update_changed=True,
            export_exact=True,
            closed_gradient_max_abs=maximum,
            atol=5e-6,
            rtol=1e-4,
            metrics=metrics,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("inputs", "gpu"))
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    globals()[args.phase](args.root.resolve())
