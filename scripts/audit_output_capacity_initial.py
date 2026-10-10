"""Compare the embedded wide model with E on six-depth GPU restoration."""

import argparse
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from flax import nnx
from lineartgen_runtime.sampling import pyramid as source_pyramid
from train_output_capacity import check

from lineartgen.diffusion import recursive_forward
from lineartgen.evaluation_sampling import runtime_inputs
from lineartgen.models.serialization import load_inference


def main(root):
    plan = check(root)
    assert jax.devices()[0].platform != "cpu"
    small, _ = load_inference(plan["initial"])
    wide, _ = load_inference(plan["warm_initial"])
    forward = nnx.jit(lambda model, x, t, s: recursive_forward(model, x, t, guides=s))
    rows = []
    with np.load(root / "inputs.npz") as data:
        for case in ("fit_straight", "fit_thin", "fit_corner", "fit_gap"):
            source = data[f"train-{case}-guides-0"]
            for partial in (False, True):
                strength = np.full_like(source, 0.1 if partial else 0.5)
                if partial:
                    strength[:, :128] = 0
                    strength[:, :, :170] = 0
                for depth in range(1, 7):
                    x, t = runtime_inputs(source, strength, 2026101044, depth)
                    guides = tuple(jax.device_put(v) for v in source_pyramid(source, depth))
                    expected = forward(small, x, t, guides)
                    actual = forward(wide, x, t, guides)
                    error = 0.0
                    for a, b, mask, guide in zip(actual, expected, t, guides, strict=True):
                        a, b, mask, guide = map(np.asarray, (a, b, mask, guide))
                        assert np.isfinite(a).all() and np.isfinite(b).all()
                        error = max(error, float(np.max(np.abs(a - b))))
                        np.testing.assert_array_equal(a[mask == 0], guide[mask == 0])
                    assert error <= plan["initial_output_max_error"], (case, partial, depth, error)
                    rows.append(dict(case=case, partial=partial, depth=depth, maximum_error=error))
    write(
        root / "initial-six-depth-audit.json",
        dict(
            rows=rows,
            conditions=len(rows),
            maximum_error=max(r["maximum_error"] for r in rows),
            tolerance=plan["initial_output_max_error"],
            all_within_tolerance=True,
            protected_exact=True,
            source_sha256=sha(root / "source.json"),
            parameter_counts={
                name: read(Path(path) / "metadata.json")["parameters"]
                for name, path in (("small", plan["initial"]), ("wide", plan["warm_initial"]))
            },
        ),
    )
    print({k: v for k, v in read(root / "initial-six-depth-audit.json").items() if k != "rows"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root.resolve())
