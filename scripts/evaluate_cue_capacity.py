"""Fixed final-checkpoint evaluation; references and train/probe are kept separate."""

import argparse
from collections import defaultdict
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from cue_capacity import EVAL_SEED, METHODS, STEPS, check, forward, prepare_inputs
from cue_preserving_data import COUNT, MODES, SIZE, inverse, strength_map
from PIL import Image, ImageDraw
from scipy import ndimage

from lineartgen.coverage_metrics import coverage_metrics
from lineartgen.evaluation import probe_metrics, protected_metrics
from lineartgen.lineart_metrics import halo_metrics
from lineartgen.models.serialization import load_inference

GROUPS = ((0, 8, 16, 24), (32, 40, 48, 56, 64), (72, 76, 80, 88))
KEYS = (
    "ink_f1_alpha50",
    "ink_deficit_fraction",
    "halo_opacity_excess",
    "coverage_overlap_f1",
    "profile_width_mae_px",
    "connection_cut_rate",
    "connection_bridge_rate",
    "blank_max_opacity",
    "black_interior_mean",
)
RULES = {
    "ink_f1_alpha50": (">=", 0.98),
    "ink_deficit_fraction": ("<=", 0.02),
    "halo_opacity_excess": ("<=", 0.003),
    "profile_width_mae_px": ("<=", 0.2),
    "connection_cut_rate": ("<=", 0),
    "connection_bridge_rate": ("<=", 0),
    "blank_max_opacity": ("<=", 0.05),
    "black_interior_mean": (">=", 0.95),
}


def metrics(prediction, data, strength, kind, mode):
    source, clean = data["source"], data["clean"]
    target = np.where(strength > 0, clean, source)
    assert np.isfinite(prediction).all()
    result = protected_metrics(prediction, target, strength)
    result |= halo_metrics(prediction, target, strength) | coverage_metrics(
        prediction, target, strength
    )
    if mode != "partial05" and len(data["probes"]):
        result |= probe_metrics(prediction, target, data["probes"], data["connections"])
    alpha = (1 - prediction[..., 0]) / 2
    if kind == "blank":
        result["blank_max_opacity"] = float(alpha.max())
    if kind == "black":
        core = (ndimage.distance_transform_edt(clean[..., 0] < -0.96) > 8) & (strength[..., 0] > 0)
        assert core.any()
        result |= dict(
            black_interior_mean=float(alpha[core].mean()),
            black_interior_min=float(alpha[core].min()),
            black_interior_pixels=int(core.sum()),
        )
    assert result["protected_max_error"] in (None, 0.0)
    assert all(v is None or np.isfinite(v) for v in result.values())
    return result


def evaluate(root):
    check(root)
    assert read(root / "snapshot-audit.json")["raw_ema_exact"]
    paths = {m: root / f"snapshots/step-{STEPS}/{m}/raw" for m in METHODS}
    models = {m: load_inference(p)[0] for m, p in paths.items()}
    out = root / "evaluation"
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    binding = {
        str(p / n): sha(p / n)
        for p in paths.values()
        for n in ("weights.npz", "config.json", "metadata.json")
    }
    write(out / "checkpoint-binding.json", binding)
    rows, refs, predictions = [], [], {}
    for record in read(root / "data.json")["records"]:
        split, index, kind = record["split"], record["index"], record["kind"]
        with np.load(root / record["file"]) as z:
            data = {k: z[k] for k in z.files}
        for mode_index, mode in enumerate(MODES):
            t = strength_map(mode)
            for name, value in (("source", data["source"]), ("inverse", inverse(data["source"]))):
                pred = np.where(t > 0, np.clip(value, -1, 1), data["source"])
                refs.append(
                    dict(
                        split=split,
                        index=index,
                        kind=kind,
                        mode=mode,
                        method=name,
                        **metrics(pred, data, t, kind, mode),
                    )
                )
            for draw in range(2):
                key = jax.random.PRNGKey(EVAL_SEED)
                for value in (int(split == "probe"), index, mode_index, draw):
                    key = jax.random.fold_in(key, value)
                p = prepare_inputs(data["clean"][None], data["source"][None], t[None], key, depth=3)
                saved = {}
                for name, model in models.items():
                    pred = np.asarray(forward(model, p)[0][0])[0]
                    rows.append(
                        dict(
                            split=split,
                            index=index,
                            kind=kind,
                            mode=mode,
                            draw=draw,
                            method=name,
                            **metrics(pred, data, t, kind, mode),
                        )
                    )
                    saved[name] = pred
                file = out / f"predictions/{split}-{index:03d}-{mode}-{draw}.npz"
                np.savez_compressed(file, **saved)
                predictions[str(file.relative_to(out))] = sha(file)
        if (index + 1) % 16 == 0:
            print(f"evaluated {split} {index + 1}/{COUNT}; model rows={len(rows)}", flush=True)
            write(out / "progress.json", dict(split=split, cases=index + 1, rows=len(rows)))
    assert len(rows) == 2304 and len(refs) == 1152
    assert {(r["split"], r["index"], r["mode"], r["draw"], r["method"]) for r in rows} == {
        (s, i, m, d, w)
        for s in ("train", "probe")
        for i in range(COUNT)
        for m in MODES
        for d in range(2)
        for w in METHODS
    }
    write(
        out / "metrics.json",
        dict(
            rows=rows,
            references=refs,
            coverage_exact=True,
            protected_exact=True,
            source_sha256=sha(root / "source.json"),
            binding_sha256=sha(out / "checkpoint-binding.json"),
        ),
    )
    write(out / "prediction-binding.json", predictions)
    summarize(root)
    render(root)
    write(
        root / "workflow-status.json", dict(stage="numerical_complete_pending_review", step=STEPS)
    )


def summarize(root):
    out = root / "evaluation"
    report = read(out / "metrics.json")
    groups = defaultdict(list)
    checks = []
    for r in report["rows"] + report["references"]:
        for kind in (r["kind"], "ALL"):
            groups[(r["split"], r["method"], r["mode"], kind)].append(r)
        for key, (operator, threshold) in RULES.items():
            value = r.get(key)
            if value is not None:
                checks.append(
                    {
                        k: r[k]
                        for k in ("split", "index", "method", "mode", "kind", "draw")
                        if k in r
                    }
                    | dict(
                        metric=key,
                        value=value,
                        threshold=threshold,
                        passed=bool(value <= threshold if operator == "<=" else value >= threshold),
                    )
                )
    summaries = []
    for g, rs in sorted(groups.items()):
        mean = {
            k: float(np.mean([r[k] for r in rs if r.get(k) is not None]))
            for k in KEYS
            if any(r.get(k) is not None for r in rs)
        }
        summaries.append(
            dict(zip(("split", "method", "mode", "kind"), g, strict=True), rows=len(rs), means=mean)
        )
    paired = []
    paired_rows = defaultdict(dict)
    for r in report["rows"]:
        paired_rows[(r["split"], r["index"], r["mode"], r["draw"])][r["method"]] = r
    for g, rs in paired_rows.items():
        small, wide = rs["width16"], rs["width32"]
        paired.append(
            dict(
                zip(("split", "index", "mode", "draw"), g, strict=True),
                differences={k: wide[k] - small[k] for k in KEYS if small.get(k) is not None},
            )
        )
    write(
        out / "summary.json",
        dict(
            groups=summaries,
            checks=checks,
            paired=paired,
            oracle_passed=all(c["passed"] for c in checks if c["method"] == "inverse"),
            adoption=False,
        ),
    )


def render(root):
    out = root / "evaluation"
    dest = out / "native-review"
    dest.mkdir(exist_ok=True)
    records = []
    for split in ("train", "probe"):
        for mode in ("whole05", "partial05"):
            for group, indices in enumerate(GROUPS):
                canvas = Image.new("RGB", (4 * SIZE, 36 + len(indices) * (SIZE + 24)), "white")
                pen = ImageDraw.Draw(canvas)
                for col, label in enumerate(("target", "source", "16 channels", "32 channels")):
                    pen.text((col * SIZE + 8, 8), label, fill="black")
                bindings = {}
                for row, index in enumerate(indices):
                    input_path = root / f"data/{split}/{index:03d}.npz"
                    pred_path = out / f"predictions/{split}-{index:03d}-{mode}-0.npz"
                    with np.load(input_path) as data, np.load(pred_path) as pred:
                        t = strength_map(mode)
                        values = (
                            np.where(t > 0, data["clean"], data["source"]),
                            data["source"],
                            pred["width16"],
                            pred["width32"],
                        )
                        for col, value in enumerate(values):
                            gray = np.rint(np.clip((value[..., 0] + 1) / 2, 0, 1) * 255).astype(
                                np.uint8
                            )
                            canvas.paste(
                                Image.fromarray(gray).convert("RGB"),
                                (col * SIZE, 60 + row * (SIZE + 24)),
                            )
                    pen.text((8, 38 + row * (SIZE + 24)), f"{split} {index} / {mode}", fill="black")
                    bindings[str(input_path)] = sha(input_path)
                    bindings[str(pred_path)] = sha(pred_path)
                file = dest / f"{split}-{mode}-{group}.png"
                canvas.save(file)
                records.append(dict(file=file.name, sha256=sha(file), bindings=bindings))
    assert len(records) == 12
    write(
        dest / "manifest.json",
        dict(records=records, native_size=SIZE, thresholding=False, contrast_rescaling=False),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("evaluate", "summarize", "render"))
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    globals()[args.action](args.root.resolve())
