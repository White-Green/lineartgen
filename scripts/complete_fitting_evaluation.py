"""Finish frozen fitting evaluations when a partial region has no teacher ink.

Inference still executes the frozen evaluator. This helper changes only summary
handling of undefined retention metrics, audits stored blocks, and records the
separate aggregation provenance. It does not change checkpoint or case selection.
"""

import argparse
from pathlib import Path

import evaluate_conditional_fitting as frozen
import numpy as np
from campaign_protocol import read, sha, write
from diagnose_learnability import verify


def fails_fit(row):
    return (
        row["ink_f1_alpha50"] < 0.95
        or (row["ink_deficit_fraction"] is not None and row["ink_deficit_fraction"] > 0.05)
        or (row["halo_opacity_excess"] or 0) > 0.02
    )


def summarize(rows, names, steps, partials):
    summaries = []
    for name in names:
        for mode in ("raw", "ema"):
            for split in ("train", "probe"):
                for count in steps:
                    for family in ("clean", "tone", "blur"):
                        for partial in partials:
                            for strength in (None, 0.1, 0.5):
                                chosen = [
                                    r
                                    for r in rows
                                    if (
                                        r["method"],
                                        r["mode"],
                                        r["split"],
                                        r["denoise_steps"],
                                        r["family"],
                                        r["partial"],
                                    )
                                    == (name, mode, split, count, family, partial)
                                    and (strength is None or r["strength"] == strength)
                                    and r["kind"] != "blank"
                                ]
                                keys = (
                                    "ink_f1_alpha50",
                                    "ink_deficit_fraction",
                                    "halo_opacity_excess",
                                    "active_mse",
                                )
                                means = {}
                                for key in keys:
                                    values = [r[key] for r in chosen if r[key] is not None]
                                    assert values, (
                                        name,
                                        mode,
                                        split,
                                        count,
                                        family,
                                        partial,
                                        strength,
                                        key,
                                    )
                                    means[key] = float(np.mean(values))
                                summaries.append(
                                    dict(
                                        method=name,
                                        mode=mode,
                                        split=split,
                                        denoise_steps=count,
                                        family=family,
                                        partial=partial,
                                        strength=strength,
                                        means=means,
                                        conditions=len(chosen),
                                        failed_fit=[
                                            dict(pair=r["pair"], strength=r["strength"])
                                            for r in chosen
                                            if fails_fit(r)
                                        ],
                                    )
                                )
    return summaries


def finish(root, step, *, reference=False):
    verify(root)
    plan = read(root / "plan.json")
    names = [plan["reference"]["method"]] if reference else plan["arms"]
    folder = root / ("evaluation/reference-1200" if reference else f"evaluation/step-{step}")
    report = read(folder / "metrics.json")
    assert report["source_sha256"] == sha(root / "source.json")
    partials = (False, True) if reference or step == 1200 else (False,)
    frozen.check_rows(report["results"], names, partials, plan["evaluation_steps"])
    for path, digest in report["artifacts"].items():
        assert sha(path) == digest, path
    summary = summarize(report["results"], names, plan["evaluation_steps"], partials)
    previous = folder / "summary.json"
    if previous.exists():
        assert read(previous) == summary
    else:
        write(previous, summary)
    if reference:
        parent = Path(plan["parent"])
        checks = []
        for mode in ("raw", "ema"):
            for split in ("train", "probe"):
                old = parent / f"evaluation/step-1200/warm_detail16-{mode}-{split}.npz"
                actual = folder / f"warm_detail16-{mode}-{split}-n1-whole.npz"
                np.testing.assert_array_equal(np.load(actual)["images"], np.load(old)["images"])
                checks.append(dict(mode=mode, split=split, exact=True, historical_sha256=sha(old)))
        write(folder / "historical-one-step-audit.json", dict(checks=checks, all_exact=True))
    missing = [
        r for r in report["results"] if r["kind"] != "blank" and r["ink_deficit_fraction"] is None
    ]
    write(
        folder / "aggregation-audit.json",
        dict(
            helper_sha256=sha(__file__),
            frozen_evaluator_sha256=sha(frozen.__file__),
            metrics_sha256=sha(folder / "metrics.json"),
            summary_sha256=sha(previous),
            nonblank_rows_without_active_teacher_ink=len(missing),
            handling="Undefined ink deficit has no retention comparison; F1 and halo still apply.",
            all_saved_artifacts_verified=True,
            predictions=report["predictions"],
            adoption=False,
        ),
    )
    print(dict(summary_complete=True, undefined_nonblank_retention_rows=len(missing)), flush=True)


def evaluate(root, step, *, reference=False):
    try:
        frozen.evaluate(root, step, reference=reference)
    except TypeError as error:
        trace = error.__traceback__
        while trace.tb_next is not None:
            trace = trace.tb_next
        line = Path(frozen.__file__).read_text().splitlines()[trace.tb_lineno - 1]
        if (
            trace.tb_frame.f_code.co_filename != frozen.__file__
            or 'r["ink_deficit_fraction"] > 0.05' not in line
            or "NoneType" not in str(error)
        ):
            raise
        print(
            "Frozen inference completed; applying audited undefined-retention summary fix.",
            flush=True,
        )
    finish(root, step, reference=reference)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("evaluate", "finish"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--step", type=int, default=1200)
    p.add_argument("--reference", action="store_true")
    args = p.parse_args()
    (evaluate if args.action == "evaluate" else finish)(
        args.root.resolve(), args.step, reference=args.reference
    )
