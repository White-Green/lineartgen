import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw

from lineartgen.config import LossConfig
from lineartgen.training.losses import image_loss, tone_weights

parser = argparse.ArgumentParser(
    description="Analytic tone-balanced MSE diagnostic; no model outputs"
)
parser.add_argument("--root", type=Path, required=True)
root = parser.parse_args().root.resolve()
if (root / "results.json").exists():
    raise FileExistsError(root / "results.json")
plan = json.loads((root / "plan.json").read_text())
source = Path(plan["data"])


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


fractions = []
for split in plan["splits"]:
    data = np.load(source / f"{split}.npz")
    meta = json.loads((source / f"{split}.json").read_text())
    for i in range(18):
        c = data["clean"][i]
        q = float(((1 - c) * 0.5).mean())
        fractions.append(
            dict(
                split=split,
                pair=i,
                kind=meta[i]["kind"],
                mean_ink=q,
                ideal_binary_black_to_white_weight=(1 - q) / q,
            )
        )
spec = plan["binary_ensemble"]
size = spec["size"]
offsets = np.asarray(spec["offsets"])
prob = np.exp(-0.5 * (offsets / spec["sigma"]) ** 2)
prob /= prob.sum()
targets = np.ones((len(offsets), size, size, 1), np.float32)
for i, offset in enumerate(offsets):
    x = spec["center"] + offset - spec["ink_width"] // 2
    targets[i, spec["row_start"] : spec["row_end"], x : x + spec["ink_width"]] = -1
weights = np.asarray(tone_weights(jnp.asarray(targets)))
ink = (1 - targets) * 0.5
posterior = np.einsum("n,nhwc->hwc", prob, ink)
numerator = np.einsum("n,nhwc->hwc", prob, weights * ink)
denominator = np.einsum("n,nhwc->hwc", prob, weights)
optima = dict(unbalanced=posterior, tone_balanced=numerator / denominator)
checks = []
for name, value in optima.items():
    config = LossConfig(balance_tones=name == "tone_balanced", edge_weight=0, contrast_weight=0)
    clean = jnp.asarray(targets)
    level = jnp.full((1, size, size, 1), plan["strength"])

    def objective(pred, config=config, clean=clean, level=level):
        per_target = jax.vmap(
            lambda target: image_loss(pred[None], target[None], level, config, 0.001)[0]
        )(clean)
        return jnp.sum(per_target * jnp.asarray(prob, jnp.float32))

    pred = jnp.asarray(1 - 2 * value, jnp.float32)
    loss, grad = jax.jit(jax.value_and_grad(objective))(pred)
    max_gradient = float(jnp.max(jnp.abs(grad)))
    assert max_gradient < 2e-7, max_gradient
    row = value[256, :, 0]
    checks.append(
        dict(
            condition=name,
            expected_loss=float(loss),
            maximum_gradient=max_gradient,
            maximum_ink=float(row.max()),
            width_above_alpha10=int((row > 0.1).sum()),
            width_above_alpha50=int((row > 0.5).sum()),
            total_cross_section_ink=float(row.sum()),
        )
    )
    np.save(root / f"{name}.npy", value)

canvas = Image.new("RGB", (5 * 128, 180), "white")
draw = ImageDraw.Draw(canvas)
images = [ink[8], ink[10], ink[12], optima["unbalanced"], optima["tone_balanced"]]
labels = [
    "teacher offset -2",
    "teacher offset 0",
    "teacher offset +2",
    "MSE optimum",
    "balanced optimum",
]
for col, (array, label) in enumerate(zip(images, labels, strict=True)):
    gray = np.rint(np.clip(1 - array[..., 0], 0, 1) * 255).astype(np.uint8)
    canvas.paste(Image.fromarray(gray[192:352, 192:320]).convert("RGB"), (col * 128, 20))
    draw.text((col * 128 + 2, 2), label, fill="black")
canvas.save(root / "analytic-native.png")
result = dict(
    plan_sha256=sha(root / "plan.json"),
    script_sha256=sha(__file__),
    loss_source_sha256=sha("src/lineartgen/training/losses.py"),
    source_data={
        f"{s}.{e}": sha(source / f"{s}.{e}") for s in plan["splits"] for e in ["json", "npz"]
    },
    actual_teacher_fractions=fractions,
    binary_ensemble_ink_fraction=float(ink[0].mean()),
    probabilities=prob.tolist(),
    checks=checks,
    learned_predictions=0,
    adoption=False,
)
(root / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(
    json.dumps(
        dict(
            checks=checks,
            teacher_ink_fraction_range=[
                min(x["mean_ink"] for x in fractions),
                max(x["mean_ink"] for x in fractions),
            ],
        ),
        indent=2,
    ),
    flush=True,
)
