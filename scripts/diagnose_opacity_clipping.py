"""CPU-only diagnostic of top-output clipping; no optimizer updates or arm selection."""

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from diagnose_learnability import SEED, fixed_inputs
from flax import nnx

from lineartgen.diffusion import recursive_forward_with_preclip
from lineartgen.models.serialization import load_inference
from lineartgen.training.opacity import opacity_deficit_loss

out = Path("runs/opacity-gradient-diagnostic-20261009").resolve()
plan = json.loads((out / "plan.json").read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


assert all(d.platform == "cpu" for d in jax.devices())
assert not (out / "results.json").exists()
forward = nnx.jit(lambda m, x, t, g: recursive_forward_with_preclip(m, x, t, guides=g))
nominal = jax.jit(jax.grad(opacity_deficit_loss))
through = jax.jit(jax.grad(lambda r, y, a: opacity_deficit_loss(jnp.clip(r, -1, 1), y, a)))
rows = []
files = {}
artifacts = {}
for entry in plan["models"]:
    root = Path(entry["root"])
    method = entry["method"]
    data = np.load(root / "probe.npz")
    checkpoint = root / f"snapshots/step-400/{method}/raw"
    for name in ["weights.npz", "config.json", "metadata.json"]:
        files[str(checkpoint / name)] = sha(checkpoint / name)
    model, _ = load_inference(checkpoint)
    refpath = root / f"evaluation/step-400/{method}-raw-probe-n1-whole.npz"
    ref = np.load(refpath)["images"]
    files[str(refpath)] = sha(refpath)
    files[str(root / "probe.npz")] = sha(root / "probe.npz")
    raws = []
    preds = []
    for case in plan["cases"]:
        i, s = case["pair"], case["strength"]
        c, r = data["clean"][i : i + 1], data["source"][i : i + 1]
        x, t, g, target, level = fixed_inputs(c, r, s, SEED + 10000 + i + 1000, partial=False)
        predictions, raws_pyramid = forward(model, x, t, g)
        pred, raw = np.asarray(predictions[0]), np.asarray(raws_pyramid[0])
        error = float(np.max(np.abs(pred[0] - ref[i + (57 if s == 0.5 else 0)])))
        assert error <= plan["cpu_gpu_max_error"], error
        core = ((1 - target) / 2 > 0.02) & (level > 0)
        truth = np.clip((1 - target) / 2, 0, 1)
        ink = np.clip((1 - pred) / 2, 0, 1)
        missing = np.maximum(truth - ink, 0)
        ng = np.asarray(nominal(pred, target, level > 0.001))
        rg = np.asarray(through(raw, target, level > 0.001))
        outside = core & (raw > 1)
        robust = core & (raw > 1 + plan["white_margin"])
        assert np.all(rg[outside] == 0)
        assert np.all(ng[outside] > 0)
        amount = float(missing[core].sum())
        den = float(truth[core].sum())
        row = dict(
            method=method,
            **case,
            core_pixels=int(core.sum()),
            cpu_gpu_max_error=error,
            ink_deficit_fraction=amount / den,
            wrong_white_core_fraction=float(outside.sum() / core.sum()),
            blocked_missing_opacity_fraction=float(missing[outside].sum() / max(amount, 1e-6)),
            robust_blocked_missing_opacity_fraction=float(
                missing[robust].sum() / max(amount, 1e-6)
            ),
            direct_gradient_positive_pixels=int((ng[core] > 0).sum()),
            through_clip_gradient_nonzero_pixels=int((rg[core] != 0).sum()),
            white_side_gradient_zero=True,
        )
        rows.append(row)
        raws.append(raw[0])
        preds.append(pred[0])
    p = out / f"{method}-cpu-preclip.npz"
    np.savez_compressed(p, raw=np.stack(raws), prediction=np.stack(preds))
    artifacts[str(p)] = sha(p)
    print(method, "complete", flush=True)
result = dict(
    plan_sha256=sha(out / "plan.json"),
    script_sha256=sha(__file__),
    files=files,
    artifacts=artifacts,
    rows=rows,
    new_updates=0,
    adoption=False,
    limitation=(
        "Only derivative from top physical preclip to this loss term. Shared-parameter changes "
        "and other pixels/losses can still recover saturated pixels. CPU diagnostic, "
        "not new GPU quality evaluation."
    ),
)
(out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
