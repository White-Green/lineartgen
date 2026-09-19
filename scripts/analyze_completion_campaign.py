"""Three-seed size comparisons; seal selection before any final-test data is opened."""

import argparse
import datetime
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np
from campaign_protocol import read, report, sha, values, write
from summarize_sketch_experiment import subset

MODELS = {
    "shared_s": ("shared", 0),
    "shared_m": ("shared", 1),
    "shared_l": ("shared", 2),
    "independent_s": ("core", 0),
    "baseline": ("baseline", 0),
}


def parameter_count(weights, *, index=None):
    """Count actual saved arrays, excluding causal extensions above a chosen prefix."""
    with np.load(weights, allow_pickle=False) as arrays:
        count = 0
        for key in arrays.files:
            path = json.loads(key)
            if index is not None:
                if path[0] == "heads" and path[1] > index:
                    continue
                if "groups" in path and path[path.index("groups") + 1] > index:
                    continue
            count += arrays[key].size
    return int(count)


def source_key(case):
    return case["source"] if case["kind"] == "real" else (case["kind"], case["variant"])


def real_ink_summaries(rows):
    """Expose nearly blank random crops separately without changing selection criteria."""
    groups = {}
    for row in rows:
        if row["strength"] <= 0 or row["suite"] not in ("matched_real", "sketch_real"):
            continue
        name = (
            "clean_real"
            if row["suite"] == "matched_real"
            else (f"sketch_real_{row['family']}_{row['severity']}")
        )
        ink = "with_ink" if row["active_target_ink_fraction"] >= 0.001 else "low_ink"
        for population in ("all", ink):
            groups.setdefault((name, population), []).append(row)
    return [
        dict(
            subset=name,
            population=population,
            cases=len(selected),
            **{
                metric: float(np.mean([r[metric] for r in selected if r.get(metric) is not None]))
                if any(r.get(metric) is not None for r in selected)
                else None
                for metric in ("active_ink_f1", "active_mae", "active_background_mae")
            },
        )
        for (name, population), selected in sorted(groups.items())
    ]


def interval(pairs, manifest, group, metric, *, seed=20260911, draws=5000):
    """Resample both fitted seeds and paired source clusters, with case weighting."""
    grouped = []
    for candidate, reference in pairs:
        baseline = {r["case"]: r for r in reference}
        groups = {}
        for row in candidate:
            if subset(row) != group:
                continue
            other = baseline[row["case"]]
            assert subset(other) == group
            a, b = row.get(metric), other.get(metric)
            if a is None and b is None:
                continue
            assert a is not None and b is not None, (group, metric, row["case"])
            cluster = source_key(manifest["cases"][row["case"]])
            groups.setdefault(cluster, []).append(a - b)
        grouped.append(groups)
    if not grouped[0]:
        assert all(not g for g in grouped)
        return None
    keys = list(grouped[0])
    assert all(set(g) == set(keys) for g in grouped)
    totals = np.array([[sum(g[k]) for k in keys] for g in grouped])
    counts = np.array([[len(g[k]) for k in keys] for g in grouped])
    random = np.random.default_rng(seed)
    si = random.integers(len(grouped), size=(draws, len(grouped), 1))
    ci = random.integers(len(keys), size=(draws, 1, len(keys)))
    sampled = totals[si, ci].sum((1, 2)) / counts[si, ci].sum((1, 2))
    return dict(
        delta=float(totals.sum() / counts.sum()),
        hierarchical_95_interval=np.quantile(sampled, [0.025, 0.975]).tolist(),
        per_seed=(totals.sum(1) / counts.sum(1)).tolist(),
        seeds=len(grouped),
        source_groups=len(keys),
        paired_cases=int(counts.sum()),
    )


def check_seal(root):
    seal = read(root / "selection-seal.json")
    for path, digest in seal["artifacts"].items():
        assert sha(path) == digest, f"Sealed selection changed: {path}"
    return seal


def original_dependencies(root):
    """Recover dependency hashes from the original snapshot, including on a fresh rerun."""
    path = root / "dependency-source.json"
    if not path.exists():
        original = read(root / "training-source.json")
        files = dict(read(root / "execution-source.json")["files"])
        assert sha(original["archive"]) == original["archive_sha256"]
        with tarfile.open(original["archive"], "r:gz") as archive:
            for member in archive:
                if (
                    member.isfile()
                    and member.name.startswith("packages/lineartgen-runtime/src/")
                    and member.name.endswith(".py")
                ):
                    files[member.name] = hashlib.sha256(
                        archive.extractfile(member).read()
                    ).hexdigest()
        write(
            path,
            dict(
                commit=original["commit"],
                file_sha256=files,
                note="Existing helper scripts and runtime dependencies from the original "
                "pre-training snapshot. New final-stage scripts are versioned separately.",
            ),
        )
    dependencies = read(path)["file_sha256"]
    for name, digest in dependencies.items():
        assert sha(name) == digest, f"Original evaluation dependency changed: {name}"
    return dependencies


def analyze(root, *, final=False):
    protocol = read(root / "protocol.json")
    cohorts = read(root / "cohorts.json")
    step = read(root / "decisions/budget.json")["step"]
    cases = root / ("final" if final else "development")
    manifest = read(cases / "cases.json")
    reports = {}
    artifacts = {str(cases / "cases.json"): sha(cases / "cases.json")}
    scope = "final-evaluation" if final else "evaluation"
    for seed, cohort in cohorts.items():
        for role, name in cohort.items():
            for mode in ("raw", "ema"):
                arm = root / "arms" / name
                reports[seed, role, mode] = report(arm, step, mode, cases, scope=scope)
                path = arm / scope / f"step-{step}-{mode}/metrics.json"
                artifacts[str(path)] = sha(path)
    rows, real_ink = [], []
    decisions = []
    for name, (role, index) in MODELS.items():
        per_mode = {}
        violations = []
        for mode in ("raw", "ema"):
            summaries = {s: values(reports[s, role, mode]) for s in cohorts}
            for seed in cohorts:
                selected_rows = [
                    r for r in reports[seed, role, mode]["results"] if r["subnet"] == index
                ]
                real_ink.extend(
                    dict(model=name, seed=seed, mode=mode, **row)
                    for row in real_ink_summaries(selected_rows)
                )
            contrasts = []
            for group in protocol["screening_subsets"]:
                for metric, tolerance in protocol["size_tolerances"].items():
                    pairs = [
                        (
                            [r for r in reports[s, role, mode]["results"] if r["subnet"] == index],
                            reports[s, "baseline", mode]["results"],
                        )
                        for s in cohorts
                    ]
                    result = interval(
                        pairs,
                        manifest,
                        group,
                        metric,
                        seed=protocol["bootstrap_seed"],
                        draws=protocol["bootstrap_draws"],
                    )
                    if result is None:
                        continue
                    lo, hi = result["hierarchical_95_interval"]
                    passes = (
                        lo >= -tolerance and min(result["per_seed"]) >= -tolerance
                        if metric == "active_ink_f1"
                        else hi <= tolerance and max(result["per_seed"]) <= tolerance
                    )
                    contrasts.append(
                        dict(
                            subset=group,
                            metric=metric,
                            tolerance=tolerance,
                            noninferior=passes,
                            **result,
                        )
                    )
                    if mode == "ema" and not passes:
                        violations.append(
                            dict(kind="noninferiority", subset=group, metric=metric, **result)
                        )
                for s in cohorts:
                    row = summaries[s].get((group, index))
                    if row is None:
                        rows.append(
                            dict(model=name, seed=s, mode=mode, subset=group, available=False)
                        )
                        if mode == "ema":
                            violations.append(dict(kind="missing_subset", seed=s, subset=group))
                        continue
                    measured = {
                        k: row.get(k) for k in ("active_ink_f1", *protocol["size_tolerances"])
                    }
                    rows.append(
                        dict(
                            model=name, seed=s, mode=mode, subset=group, available=True, **measured
                        )
                    )
                    if mode == "ema":
                        quality = protocol["absolute_quality"]
                        f1_min = (
                            quality["clean_ink_f1_min"]
                            if group.startswith("clean_")
                            else quality["strong_ink_f1_min"]
                        )
                        if row["active_ink_f1"] < f1_min:
                            violations.append(
                                dict(
                                    kind="absolute",
                                    seed=s,
                                    subset=group,
                                    metric="active_ink_f1",
                                    value=row["active_ink_f1"],
                                    minimum=f1_min,
                                )
                            )
                        for metric in (
                            "connection_cut_rate",
                            "profile_missing_fraction",
                            "connection_bridge_rate",
                        ):
                            if (
                                row.get(metric) is not None
                                and row[metric] > quality[metric + "_max"]
                            ):
                                violations.append(
                                    dict(
                                        kind="absolute",
                                        seed=s,
                                        subset=group,
                                        metric=metric,
                                        value=row[metric],
                                        maximum=quality[metric + "_max"],
                                    )
                                )
            per_mode[mode] = contrasts
        first_arm = root / "arms" / next(iter(cohorts.values()))[role]
        weights = first_arm / f"snapshots/step-{step}/model/ema/weights.npz"
        count = parameter_count(weights, index=index if role == "shared" else None)
        decisions.append(
            dict(
                model=name,
                parameters=int(count),
                eligible=not violations,
                violations=violations,
                comparisons=per_mode,
            )
        )
    eligible = sorted(
        (d for d in decisions if d["eligible"]), key=lambda d: (d["parameters"], d["model"])
    )
    result = dict(
        stage="final" if final else "development",
        step=step,
        candidate_decisions=decisions,
        per_seed_summary=rows,
        real_ink_diagnostics=real_ink,
        real_ink_definition="with_ink: active_target_ink_fraction >= 0.001; "
        "low_ink includes blank and nearly blank crops. Descriptive, not a new selection rule.",
        artifacts=artifacts,
        smallest_eligible=eligible[0]["model"] if eligible else None,
        note="Intervals resample three fitted seeds and paired sources; precision is limited. "
        "Absolute thresholds are provisional criteria, not user acceptance.",
    )
    if final:
        seal = check_seal(root)
        chosen = seal["selected_model"]
        result["sealed_selection"] = chosen
        result["selection_confirmed"] = (
            chosen is not None and next(d for d in decisions if d["model"] == chosen)["eligible"]
        )
        result["smallest_eligible"] = None
        result["final_selection_policy"] = (
            "No reselection on final test. Other fixed cohort results are diagnostic."
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=("development", "final"), required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    dependencies = original_dependencies(root)
    if args.stage == "development" and (root / "selection-seal.json").exists():
        check_seal(root)
        print("Existing selection seal verified; no reselection")
        return
    if args.stage == "final":
        check_seal(root)
    else:
        from supplement_campaign_diagnostics import run_diagnostics

        run_diagnostics(root)
    result = analyze(root, final=args.stage == "final")
    output = root / f"{args.stage}-summary.json"
    write(output, result)
    if args.stage == "development":
        artifacts = dict(result["artifacts"])
        artifacts.update(dependencies)
        for p in [
            output,
            root / "protocol.json",
            root / "cohorts.json",
            root / "training-source.json",
            root / "dependency-source.json",
            root / "diagnostic-summary.json",
            *sorted((root / "decisions").glob("*.json")),
            Path(__file__).resolve(),
            *(
                Path(__file__).resolve().parent / name
                for name in (
                    "prepare_campaign_final.py",
                    "verify_completion_campaign.py",
                    "render_campaign_blind.py",
                    "audit_completion_campaign.py",
                    "supplement_campaign_diagnostics.py",
                )
            ),
        ]:
            artifacts[str(p)] = sha(p)
        artifacts.update(read(root / "diagnostic-summary.json")["artifacts"])
        for group in read(root / "cohorts.json").values():
            for name in group.values():
                arm = root / "arms" / name
                for p in [
                    arm / "config.json",
                    arm / "variants.json",
                    *sorted((arm / f"snapshots/step-{result['step']}").rglob("*")),
                ]:
                    if p.is_file():
                        artifacts[str(p)] = sha(p)
        write(
            root / "selection-seal.json",
            dict(
                created_utc=datetime.datetime.now(datetime.UTC).isoformat(),
                selected_model=result["smallest_eligible"],
                weight_mode="ema",
                step=result["step"],
                model_cohort=list(MODELS),
                artifacts=artifacts,
            ),
        )
    print(output)


if __name__ == "__main__":
    main()
