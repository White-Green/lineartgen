"""HF exact-input pairs and position transfers after coherent broad training."""

import argparse
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import sha, write
from consistent_axis_fit import page
from cue_capacity import prepare_inputs
from flax import nnx
from learned_coarse_axis_model import infer, load_field
from plausible_study import digest
from pooled_axis_study import check

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
root = p.parse_args().root.resolve()
check(root)
out = root / "coherent-probes"
paths = {m: root / f"snapshots/step-6000/{m}/raw" for m in ("winner", "consistent")}
models = {m: load_field(path)[0] for m, path in paths.items()}
states = {m: digest(nnx.state(v, nnx.Param)) for m, v in models.items()}
records, bindings, outputs = [], {}, 0
for split in ("train", "transfer"):
    f = out / f"{split}.npz"
    bindings[str(f)] = sha(f)
    with np.load(f) as z:
        data = {k: z[k] for k in z.files}
    for depth in (1, 3, 6):
        predictions = {m: [] for m in models}
        for i, source in enumerate(data["source"]):
            t = data["strength"][i]
            prepared = prepare_inputs(
                source[None], source[None], t[None], jax.random.PRNGKey(2026101717), depth
            )
            for m, model in models.items():
                value = np.asarray(infer(model, prepared))[0]
                assert np.isfinite(value).all() and value.min() >= -1 and value.max() <= 1
                np.testing.assert_array_equal(value[t == 0], source[t == 0])
                predictions[m].append(value)
                outputs += 1
        f = out / f"{split}-depth{depth}.npz"
        np.savez_compressed(f, **{m: np.stack(v) for m, v in predictions.items()})
        bindings[str(f)] = sha(f)
        if split == "train":
            for name in models:
                for i in (0, 2, 4, 6):
                    np.testing.assert_array_equal(predictions[name][i], predictions[name][i + 1])
        for first in (0, 4):
            f = out / f"native-{split}-{depth}-{first}.png"
            ids = range(first, first + 4)
            page(
                f,
                [
                    [
                        data["source"][i],
                        data["retained_clean"][i],
                        predictions["winner"][i],
                        predictions["consistent"][i],
                    ]
                    for i in ids
                ],
                [
                    [
                        f"{split}{i} d{depth} source",
                        "retained teacher",
                        "HM control6000",
                        "HM opening6000",
                    ]
                    for i in ids
                ],
            )
            records.append(dict(file=f.name, sha256=sha(f)))
assert outputs == 96 and len(records) == 12
assert states == {m: digest(nnx.state(v, nnx.Param)) for m, v in models.items()}
for path in paths.values():
    bindings.update({str(f): sha(f) for f in path.iterdir() if f.is_file()})
write(out / "manifest.json", dict(records=records, native_scale=True))
write(
    out / "metrics.json",
    dict(outputs=outputs, states_unchanged=True, protected_exact=True, bindings=bindings),
)
print(dict(outputs=outputs, pages=len(records)), flush=True)
