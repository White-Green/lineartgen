"""Use the frozen CT diagnostics; compare a finite-family oracle at fixed capacity."""

import argparse
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from cue_capacity import prepare_inputs
from cue_preserving_data import MODES, strength_map
from evaluate_plausible import GROUPS, metrics, summarize
from path_profile_data import path_probes
from PIL import Image, ImageDraw
from plausible_data import CENTER, COUNT, SIZE, profile_samples
from precision_geometry_study import EVAL_SEED, STEPS, check
from precision_geometry_study import REPORT_METHODS as METHODS
from style_family_data import extended_data
from subpixel_field_model import infer, load_field
from wide_style_data import allstroke_candidates


def evaluate(root):
    check(root)
    assert read(root / "snapshot-audit.json")["raw_ema_exact"]

    paths = {m: root / f"snapshots/step-{STEPS}/{m}/raw" for m in METHODS} | dict(
        initial=root / "reference"
    )
    models = {m: load_field(p)[0] for m, p in paths.items()}
    out = root / "evaluation"
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    write(
        out / "checkpoint-binding.json",
        {
            str(p / n): sha(p / n)
            for p in paths.values()
            for n in ("weights.npz", "config.json", "metadata.json")
        },
    )
    rows, references, bindings = [], [], {}
    for record in read(root / "data.json")["evaluation"]:
        split, index, kind = record["split"], record["index"], record["kind"]
        with np.load(root / record["file"]) as z:
            data = {k: z[k] for k in z.files}
        probes, connections = path_probes(record["metadata"])
        full_data = data | dict(
            probes=probes,
            connections=connections,
            profiles=np.stack([profile_samples(v, probes) for v in data["variants"]]),
        )
        extended = extended_data(full_data, record)
        for mode_index, mode in enumerate(MODES):
            t = strength_map(mode)
            for reference, value in dict(
                allowed_center=data["variants"][CENTER], gray_average=data["mixture"]
            ).items():
                pred = np.where(t > 0, value, data["source"])
                references.append(
                    dict(
                        split=split,
                        index=index,
                        kind=kind,
                        mode=mode,
                        method=reference,
                        **metrics(pred, data, t, kind, mode),
                        **{
                            "full_" + k: v
                            for k, v in metrics(pred, full_data, t, kind, mode).items()
                            if k.startswith(
                                ("family_profile", "missing_profile", "profiles_", "connection_")
                            )
                        },
                    )
                )
            for draw in range(2):
                key = jax.random.PRNGKey(EVAL_SEED)
                for value in (
                    ("old_train", "new_train", "old_probe", "new_probe").index(split),
                    index,
                    mode_index,
                    draw,
                ):
                    key = jax.random.fold_in(key, value)
                prepared = prepare_inputs(
                    data["variants"][CENTER][None], data["source"][None], t[None], key, depth=3
                )
                predictions = {}
                for name, model in models.items():
                    pred = np.asarray(infer(model, prepared))[0]
                    assert np.isfinite(pred).all() and pred.min() >= -1 and pred.max() <= 1
                    np.testing.assert_array_equal(pred[t == 0], data["source"][t == 0])
                    predictions[name] = pred
                    rows.append(
                        dict(
                            split=split,
                            index=index,
                            kind=kind,
                            mode=mode,
                            draw=draw,
                            method=name,
                            **metrics(pred, data, t, kind, mode),
                            **{
                                "style_" + k: v
                                for k, v in metrics(pred, extended, t, kind, mode).items()
                                if k != "geometry_violation"
                            },
                            **{
                                "full_" + k: v
                                for k, v in metrics(pred, full_data, t, kind, mode).items()
                                if k.startswith(
                                    (
                                        "family_profile",
                                        "missing_profile",
                                        "profiles_",
                                        "connection_",
                                    )
                                )
                            },
                        )
                    )
                file = out / f"predictions/{split}-{index:03d}-{mode}-{draw}.npz"
                np.savez_compressed(file, **predictions)
                bindings[str(file.relative_to(out))] = sha(file)
        if (index + 1) % 16 == 0:
            progress = dict(split=split, cases=index + 1, model_rows=len(rows))
            write(out / "progress.json", progress)
            print(progress, flush=True)
    assert len(rows) == 4608 and len(references) == 1536
    assert {(r["split"], r["index"], r["mode"], r["draw"], r["method"]) for r in rows} == {
        (s, i, m, d, method)
        for s in ("old_train", "new_train", "old_probe", "new_probe")
        for i in range(COUNT)
        for m in MODES
        for d in range(2)
        for method in ("initial",) + METHODS
    }
    write(out / "new-metrics.json", dict(rows=rows, coverage_exact=True, protected_exact=True))
    write(
        out / "metrics.json",
        dict(
            rows=rows,
            references=references,
            coverage_exact=True,
            protected_exact=True,
        ),
    )
    write(out / "prediction-binding.json", bindings)
    summarize(root)
    summary = read(out / "summary.json")
    for group in summary["groups"]:
        values = [
            r
            for r in rows + references
            if r["split"] == group["split"]
            and r["method"] == group["method"]
            and r["mode"] == group["mode"]
            and (group["kind"] == "ALL" or r["kind"] == group["kind"])
        ]
        keys = {k for r in values for k in r if k.startswith(("full_", "style_"))}
        group["means"].update(
            {
                k: float(np.mean([r[k] for r in values if r.get(k) is not None]))
                for k in keys
                if any(r.get(k) is not None for r in values)
            }
        )
    write(out / "summary.json", summary)
    render(root)
    write(
        root / "workflow-status.json", dict(stage="numerical_complete_pending_review", step=STEPS)
    )


def render(root):
    out = root / "evaluation"
    dest = out / "native-review"
    dest.mkdir(exist_ok=True)
    records = []
    by_key = {(r["split"], r["index"]): r for r in read(root / "data.json")["evaluation"]}
    groups = [("whole05", 1, GROUPS[1]), ("partial05", 0, (6, 36, 48, 56))]
    for split in ("old_train", "new_train", "old_probe", "new_probe"):
        extra = (
            [
                ("whole05", 2, (56, 58, 59, 60)),
                ("whole05", 3, (61, 62, 63, 54)),
                ("whole05", 4, (36, 37, 38, 39)),
                ("whole05", 5, (0, 6, 12, 18)),
            ]
            if split == "new_probe"
            else []
        )
        for mode, group, indices in groups + extra:
            canvas = Image.new("RGB", (5 * SIZE, 36 + len(indices) * (SIZE + 24)), "white")
            pen = ImageDraw.Draw(canvas)
            for col, label in enumerate(
                (
                    "source",
                    "all strokes 2px",
                    "EI group6 raw18000",
                    "EN merge control",
                    "merge + 8bit",
                )
            ):
                pen.text((col * SIZE + 8, 8), label, fill="black")
            bindings = {}
            for row, index in enumerate(indices):
                ip = root / by_key[(split, index)]["file"]
                pp = out / f"predictions/{split}-{index:03d}-{mode}-0.npz"
                with np.load(ip) as data, np.load(pp) as pred:
                    t = strength_map(mode)
                    record = by_key[(split, index)]
                    options = allstroke_candidates(data, record)
                    values = (
                        data["source"],
                        np.where(t > 0, options[CENTER], data["source"]),
                        pred["initial"],
                        pred["control"],
                        pred["quant"],
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
                bindings.update({str(p): sha(p) for p in (ip, pp)})
            file = dest / f"{split}-{mode}-{group}.png"
            canvas.save(file)
            records.append(dict(file=file.name, sha256=sha(file), bindings=bindings))
    write(
        dest / "manifest.json",
        dict(records=records, native_size=SIZE, contrast_rescaling=False, thresholding=False),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    evaluate(a.root.resolve())
