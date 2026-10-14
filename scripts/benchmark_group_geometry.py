"""Preflight real-size updates for group geometry; never produces training checkpoints."""

import argparse
import time
from pathlib import Path

import group_geometry_study as study
import jax
import numpy as np
from campaign_protocol import write
from cue_capacity import make_arm
from flax import nnx
from group_field_model import GroupFieldCNN
from regional_field_objective import regional_gradient
from scale_style_balance_study import update
from subpixel_field_model import create_field_model, load_field


def main(output, methods):
    assert jax.devices()[0].platform != "cpu"
    data = study.load_data(study.PARENT)
    _, cfg = load_field(study.REFERENCE / "snapshots/step-12000/d4/raw")
    arms = {}
    for method in methods:
        c = cfg.model_copy(
            update=dict(
                model=cfg.model.model_copy(
                    update=dict(
                        channels=(study.WIDTHS[method],),
                    )
                )
            )
        )
        model = (
            create_field_model(c, seed=study.MODEL_SEED)
            if method == "plain"
            else GroupFieldCNN(c.model, rngs=nnx.Rngs(study.MODEL_SEED))
        )
        arms[method] = make_arm(c, model)
        arms[method]["gradient"] = regional_gradient()
    rows = []
    for repeat in range(2):
        for depth in range(1, 7):
            p, b, d = study.batch(data, depth - 1, "plain")
            for method in methods:
                start = time.perf_counter()
                result = update(arms[method], p, b, d)
                elapsed = time.perf_counter() - start
                assert all(np.isfinite(v) for v in result.values())
                row = dict(
                    method=method,
                    depth=depth,
                    repeat=repeat,
                    seconds=elapsed,
                    gradient_norm=result["gradient_norm"],
                )
                rows.append(row)
                print(row, flush=True)
    write(
        output,
        dict(
            rows=rows,
            device=jax.devices()[0].device_kind,
            memory=jax.devices()[0].memory_stats(),
            discarded_states=True,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=study.METHODS, default=study.METHODS)
    args = parser.parse_args()
    main(args.output, args.methods)
