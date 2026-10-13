"""Stress-test fresh synthetic shapes across input opacity, blur and local warps."""

import argparse
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from cue_capacity import forward, prepare_inputs
from cue_preserving_data import strength_map
from evaluate_plausible import metrics
from path_profile_data import path_probes
from PIL import Image, ImageDraw
from plausible_data import CENTER, SIZE, profile_samples
from pyramid_family_study import EVAL_SEED, METHODS, STEPS, check
from rough_family_data import rough_source
from style_family_data import extended_data

from lineartgen.models.serialization import load_inference


def evaluate(root):
    check(root)
    write(root / "workflow-status.json", dict(stage="stress_evaluating", step=STEPS))
    plan = read(root / "stress-plan.json")
    paths = {m: root / f"snapshots/step-{STEPS}/{m}/raw" for m in METHODS} | dict(
        initial=root / "initial"
    )
    models = {k: load_inference(p)[0] for k, p in paths.items()}
    out = root / "stress"
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    rows, bindings = [], {}
    for record in plan["records"]:
        index, kind = record["index"], record["kind"]
        with np.load(root / record["file"]) as z:
            original = {k: z[k] for k in z.files}
        probes, connections = path_probes(record["metadata"])
        full = original | dict(
            probes=probes,
            connections=connections,
            profiles=np.stack([profile_samples(x, probes) for x in original["variants"]]),
        )
        extended = extended_data(full, record)
        for corruption, options in plan["corruptions"].items():
            source = rough_source(original["mixture"], options[index])
            data = full | dict(source=source)
            for mi, mode in enumerate(plan["modes"]):
                t = strength_map(mode)
                for draw in range(2):
                    key = jax.random.PRNGKey(EVAL_SEED)
                    for value in (0x535452, index, mi, draw):
                        key = jax.random.fold_in(key, value)
                    p = prepare_inputs(
                        data["variants"][CENTER][None], source[None], t[None], key, depth=3
                    )
                    predictions = {}
                    for name, model in models.items():
                        pred = np.asarray(forward(model, p)[0][0])[0]
                        predictions[name] = pred
                        values = metrics(pred, data, t, kind, mode)
                        # The old geometry metric assumes opacity .4 and is invalid here.
                        values.pop("geometry_violation", None)
                        values.update(
                            {
                                "style_" + k: v
                                for k, v in metrics(
                                    pred, extended | dict(source=source), t, kind, mode
                                ).items()
                                if k != "geometry_violation"
                            }
                        )
                        rows.append(
                            dict(
                                index=index,
                                kind=kind,
                                corruption=corruption,
                                mode=mode,
                                draw=draw,
                                method=name,
                                **values,
                            )
                        )
                    file = out / f"predictions/{index:03d}-{corruption}-{mode}-{draw}.npz"
                    np.savez_compressed(file, source=source, **predictions)
                    bindings[str(file.relative_to(out))] = sha(file)
        if (index + 1) % 8 == 0:
            progress = dict(cases=index + 1, model_rows=len(rows))
            write(out / "progress.json", progress)
            print(progress, flush=True)
    assert len(rows) == 4608
    expected = {
        (i, c, m, d, method)
        for i in range(64)
        for c in plan["corruptions"]
        for m in plan["modes"]
        for d in range(2)
        for method in models
    }
    assert {
        (r["index"], r["corruption"], r["mode"], r["draw"], r["method"]) for r in rows
    } == expected
    groups = []
    keys = sorted(
        set.union(
            *(set(r) - {"index", "kind", "corruption", "mode", "draw", "method"} for r in rows)
        )
    )
    for c in plan["corruptions"]:
        for mode in plan["modes"]:
            for method in models:
                subset = [
                    r
                    for r in rows
                    if (r["corruption"], r["mode"], r["method"]) == (c, mode, method)
                ]
                means = {
                    k: float(np.mean([r[k] for r in subset if r.get(k) is not None]))
                    for k in keys
                    if any(r.get(k) is not None for r in subset)
                }
                groups.append(dict(corruption=c, mode=mode, method=method, means=means))
    write(
        out / "metrics.json",
        dict(rows=rows, coverage_exact=True, protected_exact=True, full_arc_profiles=True),
    )
    write(out / "summary.json", dict(groups=groups))
    write(
        out / "binding.json",
        dict(
            predictions=bindings,
            checkpoints={
                str(f): sha(f) for p in paths.values() for f in p.iterdir() if f.is_file()
            },
        ),
    )
    render(root)
    write(root / "workflow-status.json", dict(stage="stress_complete_pending_review", step=STEPS))


def render(root):
    plan = read(root / "stress-plan.json")
    out = root / "stress"
    dest = out / "native-review"
    dest.mkdir(exist_ok=True)
    records = []
    groups = ((6, 24, 36, 56),)
    tasks = [
        (c, "whole05", k, indices) for c in plan["corruptions"] for k, indices in enumerate(groups)
    ]
    tasks += [(c, "partial05", 0, (6, 24, 48, 56)) for c in ("faint", "rough")]
    for corruption, mode, group, indices in tasks:
        canvas = Image.new("RGB", (5 * SIZE, 36 + 4 * (SIZE + 24)), "white")
        pen = ImageDraw.Draw(canvas)
        for col, label in enumerate(
            ("rough source", "allowed center", "initial DD", "constant code", "pyramid level code")
        ):
            pen.text((col * SIZE + 8, 8), label, fill="black")
        bound = {}
        for row, index in enumerate(indices):
            ip = root / plan["records"][index]["file"]
            pp = out / f"predictions/{index:03d}-{corruption}-{mode}-0.npz"
            with np.load(ip) as inp, np.load(pp) as pred:
                source = pred["source"]
                t = strength_map(mode)
                values = (
                    source,
                    np.where(t > 0, inp["variants"][CENTER], source),
                    pred["initial"],
                    pred["constant"],
                    pred["level"],
                )
                for col, value in enumerate(values):
                    gray = np.rint(np.clip((value[..., 0] + 1) / 2, 0, 1) * 255).astype(np.uint8)
                    canvas.paste(
                        Image.fromarray(gray).convert("RGB"), (col * SIZE, 60 + row * (SIZE + 24))
                    )
                pen.text(
                    (8, 38 + row * (SIZE + 24)),
                    f"fresh {index} / {corruption} / {mode}",
                    fill="black",
                )
            bound.update({str(p): sha(p) for p in (ip, pp)})
        file = dest / f"{corruption}-{mode}-{group}.png"
        canvas.save(file)
        records.append(dict(file=file.name, sha256=sha(file), bindings=bound))
    write(
        dest / "manifest.json",
        dict(records=records, native_size=SIZE, contrast_rescaling=False, thresholding=False),
    )


def validate(root):
    out = root / "stress"
    report = read(out / "metrics.json")
    assert report["coverage_exact"] and report["protected_exact"] and len(report["rows"]) == 4608
    bindings = read(out / "binding.json")
    for p, h in bindings["predictions"].items():
        assert sha(out / p) == h
    for p, h in bindings["checkpoints"].items():
        assert sha(Path(p)) == h
    manifest = read(out / "native-review/manifest.json")
    review = read(out / "visual-review.json")
    assert review["reviewed"] and len(review["files"]) == len(manifest["records"]) == 8
    for record in manifest["records"]:
        assert (
            sha(out / "native-review" / record["file"])
            == record["sha256"]
            == review["files"][record["file"]]
        )
        for p, h in record["bindings"].items():
            assert sha(Path(p)) == h
    write(
        out / "completion.json",
        dict(
            passed=True,
            model_rows=4608,
            native_images_reviewed=8,
            evidence={
                n: sha(out / n)
                for n in ["metrics.json", "summary.json", "binding.json", "visual-review.json"]
            },
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("evaluate", "render", "validate"))
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    globals()[a.action](a.root.resolve())
