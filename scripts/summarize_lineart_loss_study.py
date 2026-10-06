"""Audit fixed study outputs and report paired source-cluster differences."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from campaign_protocol import read, sha, write
from summarize_sketch_experiment import clustered_delta
from train_lineart_loss_study import verify

from lineartgen.training.lineart_study import METHODS

METRICS = (
    "halo_opacity_excess",
    "halo_visible_fraction",
    "halo_gray_fraction",
    "far_opacity_excess",
    "ink_deficit_fraction",
    "ink_f1_alpha10",
    "ink_f1_alpha25",
    "ink_f1_alpha50",
    "ink_f1_alpha75",
    "active_ink_f1",
    "active_mae",
    "profile_width_mae_px",
    "profile_missing_fraction",
    "profile_offset_mae_px",
    "profile_offset_variation_px",
    "connection_cut_rate",
    "connection_bridge_rate",
    "edit_boundary_mae",
)


def groups(row):
    primary = "/".join(row[key] for key in ("domain", "family", "severity"))
    yield primary
    yield primary + ("/partial" if row["partial"] else "/whole")
    yield primary + f"/strength-{row['strength']:g}"
    if row["domain"] == "geometry":
        yield primary + "/shape-" + row["kind"]


def collect(rows):
    result = defaultdict(list)
    for row in rows:
        for group in groups(row):
            result[group, row["method"], row["denoise_steps"]].append(row)
    return result


def summarize(rows):
    return [
        dict(group=group, method=method, denoise_steps=steps, cases=len(items))
        | {
            key: float(np.mean([r[key] for r in items if r.get(key) is not None]))
            if any(r.get(key) is not None for r in items)
            else None
            for key in METRICS
        }
        for (group, method, steps), items in sorted(collect(rows).items())
    ]


def differences(rows, reference, manifest, baseline):
    lookup = {(r["case"], r["denoise_steps"]): r for r in reference if r["method"] == baseline}
    paired = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["method"] == baseline:
            continue
        before = lookup[row["case"], row["denoise_steps"]]
        case = manifest["cases"][row["case"]]
        # Legacy shapes do not have variants: conservatively treat each kind as one cluster.
        cluster = (
            case["source"]
            if case["kind"] == "real"
            else (case["kind"], case.get("variant", "legacy"))
        )
        for metric in METRICS:
            if row.get(metric) is None or before.get(metric) is None:
                continue
            for group in groups(row):
                paired[group, row["method"], row["denoise_steps"], metric][cluster].append(
                    row[metric] - before[metric]
                )
    return [
        dict(group=group, method=method, denoise_steps=steps, metric=metric, baseline=baseline)
        | clustered_delta(values)
        for (group, method, steps, metric), values in sorted(paired.items())
    ]


def audit(root, step, mode, reference, count):
    path = root / f"evaluation/step-{step}-{mode}/metrics.json"
    report = read(path)
    assert report["step"] == step and report["mode"] == mode
    assert report["preflight_limit"] is None and not report["diagnostic"]
    assert report["manifest_sha256"] == reference["cases_sha256"]
    assert report["recursion_levels"] == 3 and report["noise"] == "fixed NumPy runtime draws"
    names = ("initial",) if step == 0 else METHODS
    expected = {(i, name, 1) for i in range(count) for name in names}
    actual = [(r["case"], r["method"], r["denoise_steps"]) for r in report["results"]]
    assert len(actual) == len(expected) and set(actual) == expected
    assert report["prediction_count"] == len(expected)
    for row in report["results"]:
        assert row["protected_max_error"] in (None, 0.0)
        assert all(np.isfinite(v) for v in row.values() if isinstance(v, (int, float)))
    protocol = read(root / "protocol.json")
    for name in names:
        folder = (
            Path(protocol["initial"])
            if step == 0
            else root / "snapshots" / f"step-{step}" / name / mode
        )
        assert sha(folder / "weights.npz") == report["model_weights_sha256"][name]
    return report, {str(path): sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--through", choices=(0, 2000, 10000), type=int, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    protocol = verify(root)
    reference = read(root / "evaluation.json")
    manifest_path = Path(reference["root"]) / "cases.json"
    assert sha(manifest_path) == reference["cases_sha256"]
    manifest = read(manifest_path)
    initial, artifacts = audit(root, 0, "ema", reference, len(manifest["cases"]))
    output = dict(initial=dict(summary=summarize(initial["results"])), milestones={})
    for step in [n for n in protocol["milestones"] if n <= args.through]:
        for mode in ("ema", "raw"):
            report, files = audit(root, step, mode, reference, len(manifest["cases"]))
            artifacts.update(files)
            rows = report["results"]
            output["milestones"][f"{step}-{mode}"] = dict(
                summary=summarize(rows),
                paired=[
                    r
                    for baseline in ("control", "no_contrast")
                    for r in differences(rows, rows, manifest, baseline)
                ]
                + differences(rows, initial["results"], manifest, "initial"),
            )
    write(
        root / f"analysis-through-{args.through}.json",
        dict(
            through=args.through,
            **output,
            artifacts=artifacts,
            analysis_source_sha256=sha(Path(__file__)),
            limitations="Synthetic sketches, one training seed, reused development data. "
            "Intervals resample source-image or geometry-kind/variant clusters and describe "
            "uncertainty conditional on these fitted models, not training-seed uncertainty. "
            "Multiple exploratory groups/metrics; intervals are not corrected for selection. "
            "No automatic quality acceptance; visual review remains required.",
        ),
    )
    primary = output["initial"]["summary"]
    if args.through:
        primary = output["milestones"][f"{args.through}-ema"]["summary"]
    lines = [
        f"# Lineart objective study, additional step {args.through}",
        "",
        "EMA, runtime 3 levels / 1 denoise step. Lower errors are better; higher F1 is better.",
        "Synthetic development only. Gray reduction alone does not establish quality improvement.",
        "",
        "| Group | Method | N | Excess halo opacity | Ink deficit | Alpha .5 F1 | "
        "Missing profiles | Cuts | Bridges |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in primary:
        if row["group"].count("/") != 2:
            continue
        values = [
            row[k]
            for k in (
                "halo_opacity_excess",
                "ink_deficit_fraction",
                "ink_f1_alpha50",
                "profile_missing_fraction",
                "connection_cut_rate",
                "connection_bridge_rate",
            )
        ]
        cells = ["—" if value is None else f"{value:.5f}" for value in values]
        lines.append(
            f"| {row['group']} | {row['method']} | {row['cases']} | " + " | ".join(cells) + " |"
        )
    (root / f"analysis-through-{args.through}.md").write_text("\n".join(lines) + "\n")
    print(f"Audited reports through additional step {args.through}")


if __name__ == "__main__":
    main()
