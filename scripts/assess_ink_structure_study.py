"""Apply prespecified halo/line-retention gates, without replacing visual review."""

import argparse
from pathlib import Path

from campaign_protocol import read, sha, write

from lineartgen.training.lineart_study import study_methods


def assess(rows, rules, method, *, screening=False):
    lookup = {(r["group"], r["method"]): r for r in rows}
    baseline = rules["primary_baseline"]
    gains, failures, comparisons = {}, [], []
    for group in rules["halo_groups"]:
        a, b = lookup[group, method], lookup[group, baseline]
        before, after = b["halo_opacity_excess"], a["halo_opacity_excess"]
        if before is None or after is None or before <= 0:
            raise ValueError(f"Undefined primary halo: {group}")
        gains[group] = (before - after) / before
    gains_ok = (
        max(gains.values()) >= 0.05
        if screening
        else min(gains.values()) >= rules["minimum_relative_halo_gain"]
    )
    if not gains_ok:
        failures.append(dict(reason="insufficient_halo_gain", gains=gains))
    tolerances = (
        {
            "ink_deficit_fraction": 0.02,
            "profile_missing_fraction": 0.03,
            "connection_cut_rate": 0.03,
            "connection_bridge_rate": 0.03,
        }
        if screening
        else rules["maximum_increases"]
    )
    for primary in rules["safety_groups"]:
        for suffix in ["primary"] if screening else rules["required_strata"]:
            group = primary if suffix == "primary" else primary + "/" + suffix
            if (group, method) not in lookup or (group, baseline) not in lookup:
                raise ValueError(f"Required stratum absent: {group}, {method}")
            a, b = lookup[group, method], lookup[group, baseline]
            for metric, limit in tolerances.items():
                before, after = b[metric], a[metric]
                if before is None and after is None:
                    continue  # Geometry-only probes and wholly protected regions.
                if before is None or after is None:
                    raise ValueError(f"Mismatched defined metrics: {group}, {metric}")
                row = dict(group=group, metric=metric, delta=after - before, tolerance=limit)
                comparisons.append(row)
                if row["delta"] > limit:
                    failures.append(row)
            if not screening:
                before, after = b["active_ink_f1"], a["active_ink_f1"]
                if before is None or after is None:
                    raise ValueError(f"Undefined required F1: {group}")
                row = dict(
                    group=group,
                    metric="active_ink_f1_drop",
                    delta=before - after,
                    tolerance=rules["maximum_active_f1_drop"],
                )
                comparisons.append(row)
                if row["delta"] > row["tolerance"]:
                    failures.append(row)
    return dict(
        method=method,
        baseline=baseline,
        screening=screening,
        numerical_gate_passed=not failures,
        primary_halo_gains=gains,
        failures=failures,
        comparisons=comparisons,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    analysis = root / f"analysis-through-{args.step}.json"
    data, protocol, rules = (
        read(analysis),
        read(root / "protocol.json"),
        read(root / "acceptance.json"),
    )
    for path, digest in data["artifacts"].items():
        assert sha(path) == digest, path
    rows = data["milestones"][f"{args.step}-ema"]["summary"]
    methods = [m for m in study_methods(protocol) if m not in ("control", "no_contrast")]
    results = [assess(rows, rules, m, screening=args.step == 2000) for m in methods]
    ranking = sorted(
        [r for r in results if r["numerical_gate_passed"]],
        key=lambda r: -sum(r["primary_halo_gains"].values()),
    )
    write(
        root / f"quality-gate-{args.step}.json",
        dict(
            step=args.step,
            results=results,
            ranked_provisional_candidates=[r["method"] for r in ranking],
            analysis_sha256=sha(analysis),
            rules_sha256=sha(root / "acceptance.json"),
            script_sha256=sha(Path(__file__)),
            meaning="Provisional numerical gate only; visual review, fresh conditions and another "
            "training seed are still required before adopting a model.",
        ),
    )
    print([(r["method"], r["numerical_gate_passed"], len(r["failures"])) for r in results])


if __name__ == "__main__":
    main()
