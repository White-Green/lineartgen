"""Paired adjacent augmentation factors and width differences at the final checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path

from summarize_sketch_experiment import clustered_delta, subset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "evaluation/cases.json").read_text())
    primary = json.loads(args.primary.read_text())
    records, hashes = [], {}
    for mode in ("raw", "ema"):
        path = root / "evaluation" / f"step-10000-{mode}" / "metrics.json"
        report = json.loads(path.read_text())
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        groups = {}
        for row in report["results"]:
            name = subset(row)
            if name is not None and row["denoise_steps"] == 1:
                groups.setdefault((name, row["method"], row["subnet"]), {})[row["case"]] = row
        pairs = [
            ((candidate, width), (reference, width))
            for candidate, reference in (("tone", "control"), ("blur", "tone"), ("strokes", "blur"))
            for width in range(3)
        ]
        pairs += [(("strokes", a), ("strokes", b)) for a, b in ((1, 0), (2, 1), (2, 0))]
        for name in sorted({k[0] for k in groups}):
            for candidate, baseline in pairs:
                left, right = groups[name, *candidate], groups[name, *baseline]
                assert left.keys() == right.keys()
                metrics = ["active_ink_f1", "active_mae"]
                if "geometry" in name:
                    metrics += [
                        "profile_width_mae_px",
                        "connection_cut_rate",
                        "connection_bridge_rate",
                    ]
                for metric in metrics:
                    differences = {}
                    for index, row in left.items():
                        ref = right[index]
                        if row.get(metric) is None or ref.get(metric) is None:
                            continue
                        case = manifest["cases"][index]
                        cluster = (
                            case["source"]
                            if case["kind"] == "real"
                            else (case["kind"], case["variant"])
                        )
                        differences.setdefault(cluster, []).append(row[metric] - ref[metric])
                    delta = clustered_delta(differences)
                    if delta is None:
                        continue
                    record = dict(
                        mode=mode,
                        subset=name,
                        method=candidate[0],
                        subnet=candidate[1],
                        baseline=baseline[0],
                        baseline_subnet=baseline[1],
                        metric=metric,
                        denoise_steps=1,
                        **delta,
                    )
                    # The overlapping tone/control comparisons must reproduce the primary report.
                    if baseline[0] == "control":
                        expected = next(
                            r
                            for r in primary["evaluations"][f"10000-{mode}"]["paired_deltas"]
                            if r["subset"] == name
                            and r["method"] == candidate[0]
                            and r["subnet"] == candidate[1]
                            and r["baseline"] == "control"
                            and r["metric"] == metric
                            and r["denoise_steps"] == 1
                        )
                        assert all(expected[k] == v for k, v in delta.items())
                    records.append(record)
    args.output.write_text(
        json.dumps(
            dict(
                checkpoint_step=10000,
                records=records,
                evaluation_sha256=hashes,
                primary_sha256=hashlib.sha256(args.primary.read_bytes()).hexdigest(),
                note="Case-weighted paired differences; 5000 source-cluster bootstrap draws, "
                "seed 20260911. One training seed. Intervals are conditional on fitted models "
                "and this synthetic evaluation.",
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    print(f"Saved {len(records)} paired contrasts", flush=True)


if __name__ == "__main__":
    main()
