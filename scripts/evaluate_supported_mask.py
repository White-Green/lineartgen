"""Fixed HC pair and mask/position transfers after broad training."""

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
out = root / "mask-probes"
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
                source[None], source[None], t[None], jax.random.PRNGKey(2026101714), depth
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
        f = out / f"native-{split}-{depth}.png"
        page(
            f,
            [
                [
                    data["source"][i],
                    data["global_clean"][i],
                    data["local_clean"][i],
                    predictions["winner"][i],
                    predictions["consistent"][i],
                ]
                for i in range(len(data["source"]))
            ],
            [
                [
                    f"{split}{i} d{depth} source",
                    "old permitted",
                    "support permitted",
                    "HH retained6000",
                    "HI learned coarse6000",
                ]
                for i in range(len(data["source"]))
            ],
        )
        records.append(dict(file=f.name, sha256=sha(f)))
assert outputs == 36 and len(records) == 6
assert states == {m: digest(nnx.state(v, nnx.Param)) for m, v in models.items()}
for path in paths.values():
    bindings.update({str(f): sha(f) for f in path.iterdir() if f.is_file()})
write(out / "manifest.json", dict(records=records, native_scale=True))
write(
    out / "metrics.json",
    dict(outputs=outputs, states_unchanged=True, protected_exact=True, bindings=bindings),
)
print(dict(outputs=outputs, pages=6), flush=True)
