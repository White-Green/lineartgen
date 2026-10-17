"""Prepare GP batches plus GV ambiguity, preserving whole-only axis permission.

This module does not start training. A study must bind the referenced data and
fix its model, loss mode and schedule before using these batches.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from ambiguous_shape_data import record, render, selected
from axis_family_data import axis_payload
from campaign_protocol import read
from plausible_study import digest, exact
from pooled_procedural_study import GM
from pooled_procedural_study import batch as gp_batch
from pooled_procedural_study import load_data as gp_load_data
from procedural_shape_data import selected as long_selected
from procedural_shape_study import procedural_inputs, select

GV = Path("../jax-ambiguous-shape-data-20261017/runs/ambiguous-shape-data-20261017").resolve()


def load_data():
    assert read(GV / "completion.json")["stage"] == "complete_ambiguous_shape_data_preview"
    assert read(GV / "decision.json")["teacher_preview_passed"]
    data = gp_load_data()
    data["ambiguous_recipes"] = {(r["step"], r["item"]): r for r in read(GV / "recipes.json")}
    data["gp_rows"] = [
        json.loads(s)
        for s in (
            Path(
                "../jax-pooled-procedural-20261017/runs/pooled-procedural-20261017-retry1/training/sampling.jsonl"
            )
        )
        .read_text()
        .splitlines()
    ]
    data["axis_widths"] = read(GM / "width-plan.json")
    return data


def batch(data, step):
    if data.get("cached_pooled_axis_step") == step:
        data["last_audit"] = data["cached_pooled_axis_audit"]
        return data["cached_pooled_axis_batch"]
    p, c, d = gp_batch(data, step)
    gp = data["gp_rows"][step]
    assert digest((p, c[0], d)) == gp["input_sha256"]
    assert digest(c[:3]) == gp["candidate_payload_sha256"]
    assert np.asarray(c[3]).tolist() == gp["context_keep"]
    audit = dict(
        data["last_audit"],
        ambiguous_selected=selected(step),
        original_gp_input_sha256=gp["input_sha256"],
        original_gp_candidate_sha256=gp["candidate_payload_sha256"],
    )
    if selected(step):
        assert not long_selected(step)
        records = [record(step, i) for i in range(4)]
        assert records == [data["ambiguous_recipes"][step, i] for i in range(4)]
        arrays = [render(r) for r in records]
        source, clean, base, directions, axis, axis_directions = (
            jnp.asarray(np.stack([a[k] for a in arrays]))
            for k in ("source", "clean", "base", "direction", "axis_base", "axis_direction")
        )
        prepared = procedural_inputs(
            clean,
            source,
            p.noise_levels,
            p.progress,
            jax.random.fold_in(jax.random.PRNGKey(2026101707), step),
        )
        exact(prepared.noise_levels, p.noise_levels)
        exact(prepared.progress, p.progress)
        allowed = jnp.all(p.noise_levels[0] > 0, axis=(1, 2, 3))
        payload = (base, base, jnp.zeros(4, dtype=bool), c[3], axis, axis_directions, allowed)
        result = prepared, payload, directions
        audit |= dict(
            ambiguous_records=records,
            ambiguous_arrays_sha256=digest(
                (source, clean, base, directions, axis, axis_directions)
            ),
            alternate_allowed=np.asarray(allowed).tolist(),
            family="ambiguous",
        )
    elif long_selected(step):
        # GL's long bars and fills must not inherit unrelated legacy geometry.
        result = p, (*c, c[0], d, jnp.zeros(4, dtype=bool)), d
        audit |= dict(alternate_allowed=[False] * 4, family="long_bar_fill")
    else:
        axis, records = axis_payload(data, select(step), data["axis_widths"][step]["records"], c, d)
        assert all(not r["allowed"] or select(step)["modes"][r["item"]] != 2 for r in records)
        result = p, (*c, *axis), d
        audit |= dict(
            axis_records=records,
            alternate_allowed=np.asarray(axis[2]).tolist(),
            family="legacy",
        )
    data["last_audit"] = data["cached_pooled_axis_audit"] = audit
    data["cached_pooled_axis_step"] = step
    data["cached_pooled_axis_batch"] = result
    return result
