"""GPU forward comparison of zero source projections with their exact 100k parent."""

import argparse
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from flax import nnx

from lineartgen.config import load_config
from lineartgen.evaluation_sampling import inference_with_source, runtime_inputs
from lineartgen.models import create_model
from lineartgen.models.serialization import load_inference
from lineartgen.training.lineart_study import assert_initial_parameters


def audit(root):
    assert all("cpu" not in str(d).lower() for d in jax.devices())
    initial, _ = load_inference(read(root / "protocol.json")["initial"])
    config = load_config(root / "config.json")
    model = create_model(config.model, rngs=nnx.Rngs(config.train.initialization_seed))
    nnx.update(model, nnx.state(initial, nnx.Param))
    assert_initial_parameters(initial, model)
    ref = read(root / "evaluation.json")
    cases = Path(ref["root"])
    assert sha(cases / "cases.json") == ref["cases_sha256"]
    manifest = read(cases / "cases.json")
    infer = nnx.jit(
        lambda m, x, t, s, n: inference_with_source(m, x, t, s, steps=n), static_argnums=4
    )
    rows = []
    for case in [0, 480, 903, 1295]:
        item = manifest["cases"][case]
        p = cases / item["file"]
        assert sha(p) == ref["files"][item["file"]]
        with np.load(p, allow_pickle=False) as data:
            if "source" not in data:
                assert item.get("family", "clean") == "clean"
            source = data["source"] if "source" in data else data["image"]
            strength = data["strength"]
        for levels in range(1, 7):
            inputs = jax.device_put(
                runtime_inputs(source[None], strength[None], item["seed"], levels)
            )
            for steps in [1, 2, 5] if levels == 3 else [1]:
                old = np.asarray(infer(initial, *inputs, jax.device_put(source[None]), steps))
                new = np.asarray(infer(model, *inputs, jax.device_put(source[None]), steps))
                assert np.isfinite(old).all() and np.isfinite(new).all()
                np.testing.assert_allclose(new, old, rtol=2e-5, atol=2e-6)
                rows.append(
                    dict(
                        case=case,
                        levels=levels,
                        steps=steps,
                        exact=np.array_equal(old, new),
                        max_error=float(np.max(np.abs(old - new))),
                    )
                )
        print(dict(case=case, completed=len(rows)), flush=True)
    write(
        root / "initial-guide-audit.json",
        dict(
            rows=rows,
            devices=[str(d) for d in jax.devices()],
            all_exact=all(r["exact"] for r in rows),
            atol=2e-6,
            rtol=2e-5,
            inherited_parameters_exact=True,
            guide_parameters_initially_zero=True,
            script_sha256=sha(Path(__file__)),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    audit(p.parse_args().root.resolve())
