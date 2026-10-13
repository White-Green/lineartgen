"""Use the frozen CT diagnostics; compare a finite-family oracle at fixed capacity."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from cue_capacity import forward, prepare_inputs
from cue_preserving_data import MODES, strength_map
from evaluate_plausible import GROUPS, metrics, summarize
from PIL import Image, ImageDraw
from plausible_data import CENTER, COUNT, SIZE
from robust_family_study import EVAL_SEED, METHODS, SAVED, STEPS, check

from lineartgen.models.serialization import load_inference


def evaluate(root):
    check(root)
    assert read(root / "snapshot-audit.json")["raw_ema_exact"]

    paths = {m: root / f"snapshots/step-{STEPS}/{m}/raw" for m in METHODS} | dict(
        initial=root / "initial"
    )
    models = {m: load_inference(p)[0] for m, p in paths.items()}
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
                    pred = np.asarray(forward(model, prepared)[0][0])[0]
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
    groups = [("whole05", n, indices) for n, indices in enumerate(GROUPS)] + [
        ("partial05", 0, (24, 36, 48, 56))
    ]
    for split in ("old_train", "new_train", "old_probe", "new_probe"):
        for mode, group, indices in groups:
            canvas = Image.new("RGB", (6 * SIZE, 36 + len(indices) * (SIZE + 24)), "white")
            pen = ImageDraw.Draw(canvas)
            for col, label in enumerate(
                (
                    "source",
                    "allowed center",
                    "gray average",
                    "initial CV",
                    "dynamic Charb",
                    "stable Charb",
                )
            ):
                pen.text((col * SIZE + 8, 8), label, fill="black")
            bindings = {}
            for row, index in enumerate(indices):
                ip = root / by_key[(split, index)]["file"]
                pp = out / f"predictions/{split}-{index:03d}-{mode}-0.npz"
                with np.load(ip) as data, np.load(pp) as pred:
                    t = strength_map(mode)
                    values = (
                        data["source"],
                        np.where(t > 0, data["variants"][CENTER], data["source"]),
                        np.where(t > 0, data["mixture"], data["source"]),
                        pred["initial"],
                        pred["dynamic"],
                        pred["stable"],
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


def finalize(root):
    check(root)
    tests = read(root / "test-results.json")
    assert tests["passed"]
    for p, h in tests["logs"].items():
        assert sha(root / p) == h
    assert read(root / "initial-audit.json")["six_depth_outputs_exact"]
    assert read(root / "backend.json")["flags"] == ""
    prefix = read(root / "prefix-audit.json")
    assert prefix["passed"] and len(prefix["metrics"]) == 6
    for audit in [prefix["states"], *prefix["metrics"]]:
        assert set(audit) == set(METHODS)
        assert all(x["passed"] and x["atol"] == 1e-7 and x["rtol"] == 1e-5 for x in audit.values())
    for step in SAVED[1:]:
        assert read(root / f"training/restore-{step}.json")["full_state_exact"]
    snapshots = read(root / "snapshot-audit.json")
    assert snapshots["raw_ema_exact"] and snapshots["exports"] == 12
    for p, h in snapshots["bindings"].items():
        assert sha(Path(p)) == h
    rows = [json.loads(s) for s in (root / "training/sampling.jsonl").read_text().splitlines()]
    assert [x["step"] for x in rows] == list(range(1, STEPS + 1))
    for r, p in zip(rows, read(root / "sampling-plan.json"), strict=True):
        assert all(r[k] == p[k] for k in ("indices", "variants", "modes", "depth"))
        assert set(r["metrics"]) == set(METHODS)
    assert len({x["input_sha256"] for x in rows}) == STEPS
    assert all(len(set(x["per_arm_input_sha256"].values())) == 1 for x in rows)
    out = root / "evaluation"
    report = read(out / "new-metrics.json")
    assert report["coverage_exact"] and report["protected_exact"] and len(report["rows"]) == 4608
    for p, h in read(out / "prediction-binding.json").items():
        assert sha(out / p) == h
    for p, h in read(out / "checkpoint-binding.json").items():
        assert sha(Path(p)) == h
    manifest = read(out / "native-review/manifest.json")
    review = read(out / "visual-review.json")
    assert review["reviewed"] and len(review["files"]) == len(manifest["records"]) == 16
    for record in manifest["records"]:
        assert (
            sha(out / "native-review" / record["file"])
            == record["sha256"]
            == review["files"][record["file"]]
        )
        for p, h in record["bindings"].items():
            assert sha(Path(p)) == h
    write(
        root / "completion.json",
        dict(
            stage="complete_robust_family",
            adoption=False,
            oracle_diagnostic=True,
            numerical_commit=read(root / "source.json")["commit"],
            steps_per_arm=STEPS,
            methods=METHODS,
            model_rows=4608,
            native_images_reviewed=16,
            checkpoint_restore_exact=True,
            raw_ema_exports_exact=True,
            evidence={
                n: sha(root / n)
                for n in (
                    "source.json",
                    "plan.json",
                    "test-results.json",
                    "prefix-audit.json",
                    "snapshot-audit.json",
                    "training/sampling.jsonl",
                    "evaluation/new-metrics.json",
                    "evaluation/summary.json",
                    "evaluation/visual-review.json",
                )
            },
        ),
    )
    write(root / "workflow-status.json", dict(stage="complete_robust_family", step=STEPS))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("evaluate", "render", "finalize"))
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    globals()[a.action](a.root.resolve())
