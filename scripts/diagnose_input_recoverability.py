"""Known-corruption inverse diagnostic for the frozen local fitting corpus."""

import json
from pathlib import Path

import numpy as np
from campaign_protocol import read, sha
from scipy.linalg import eigh, matmul_toeplitz
from scipy.ndimage import convolve1d

from lineartgen.lineart_metrics import halo_metrics

r = Path("runs/detail-head-fitting-20261008")
out = r / "input-recoverability"
plan = read(out / "plan.json")
axis = np.arange(-6, 7, dtype=np.float32)
w = np.exp(-0.5 * (axis / 2) ** 2)
w /= w.sum()
# Symmetric finite Toeplitz convolution, including zero-padded boundaries.
c = np.zeros(512, np.float64)
c[:7] = w[6:]
A = matmul_toeplitz((c, c), np.eye(512))
d, U = eigh(A, check_finite=True)
transfer = d[:, None] * d[None, :]
rows = []
checks = []
counts = []
for split in plan["splits"]:
    data = np.load(r / f"{split}.npz")
    meta = read(r / f"{split}.json")
    targets = data["clean"][38:57]
    sources = data["source"][38:57]
    predictions = {
        f"{variant}-{lam}": [] for variant in plan["inputs"] for lam in plan["regularization"]
    }
    for i, (target, source) in enumerate(zip(targets, sources, strict=True)):
        ink = (1 - target[..., 0].astype(np.float64)) * 0.5
        forward = convolve1d(
            convolve1d(ink, w.astype(np.float64), axis=0, mode="constant"),
            w.astype(np.float64),
            axis=1,
            mode="constant",
        )
        error = float(np.max(np.abs(1 - 2 * 0.3 * forward - source[..., 0])))
        assert error < 1e-6, error
        checks.append(dict(split=split, pair=38 + i, max_forward_error=error))
        alpha = np.rint((1 - source[..., 0]) * 0.5 * 255).astype(np.uint8)
        variants = {
            "float": source[..., 0].astype(np.float64),
            "alpha8": 1 - 2 * alpha.astype(np.float64) / 255,
            "alpha8_scale1.1": np.clip((1 - 2 * alpha.astype(np.float64) / 255) * 1.1, -1, 1),
        }
        for variant, s in variants.items():
            counts.append(
                dict(
                    split=split,
                    pair=38 + i,
                    kind=meta[38 + i]["kind"],
                    variant=variant,
                    nonzero_pixels=int(np.count_nonzero(s < 1)),
                    maximum_input_ink=float(((1 - s) * 0.5).max()),
                    blank=bool(np.all(s == 1)),
                )
            )
            # Divide the known normalization scale. Clipped white has unknown original
            # alpha; assign zero ink there, retaining this limitation explicitly.
            observed = (
                (1 - s) * 0.5
                if variant != "alpha8_scale1.1"
                else np.where(s < 1, (1 - s / 1.1) * 0.5, 0)
            )
            transformed = U.T @ (observed / 0.3) @ U
            for lam in plan["regularization"]:
                restored = U @ (transformed * transfer / (transfer**2 + lam)) @ U.T
                pred = (1 - 2 * np.clip(restored, 0, 1)).astype(np.float32)[..., None]
                metrics = halo_metrics(pred, target, np.ones_like(target))
                rows.append(
                    dict(
                        split=split,
                        pair=38 + i,
                        kind=meta[38 + i]["kind"],
                        variant=variant,
                        regularization=lam,
                        **metrics,
                    )
                )
                predictions[f"{variant}-{lam}"].append(pred)
    for key, values in predictions.items():
        np.savez_compressed(out / f"{split}-{key}.npz", images=np.stack(values))
summary = []
for split in plan["splits"]:
    for variant in plan["inputs"]:
        for lam in plan["regularization"]:
            selected = [
                x
                for x in rows
                if x["split"] == split
                and x["variant"] == variant
                and x["regularization"] == lam
                and x["kind"] != "blank"
            ]
            summary.append(
                dict(
                    split=split,
                    variant=variant,
                    regularization=lam,
                    means={
                        k: float(np.mean([x[k] for x in selected]))
                        for k in ["ink_f1_alpha50", "halo_opacity_excess", "ink_deficit_fraction"]
                    },
                )
            )
result = dict(
    plan_sha256=sha(out / "plan.json"),
    script_sha256=sha(Path(__file__)),
    source_data={
        f"{split}.{ext}": sha(r / f"{split}.{ext}")
        for split in plan["splits"]
        for ext in ("json", "npz")
    },
    forward_checks=checks,
    input_counts=counts,
    results=rows,
    summary=summary,
    adoption=False,
)
(out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
for row in summary:
    print(row, flush=True)
