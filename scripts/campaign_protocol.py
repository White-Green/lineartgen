"""Frozen decision rules and completeness checks for the remaining experiments."""

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from summarize_sketch_rates import contrast, summaries


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def report(arm, step, mode, cases, *, scope="evaluation"):
    path = Path(arm) / scope / f"step-{step}-{mode}/metrics.json"
    result = read(path)
    count = len(read(Path(cases) / "cases.json")["cases"])
    config = read(Path(arm) / "config.json")
    widths = 1 + len(config["model"]["nested_channels"])
    expected = set(itertools.product(range(count), ("model",), range(widths)))
    actual = [(r["case"], r["method"], r["subnet"]) for r in result["results"]]
    assert len(actual) == len(expected) and set(actual) == expected, path
    assert result["checkpoint_step"] == step and result["ema"] == (mode == "ema")
    assert result["cases"] == count and result["manifest_sha256"] == sha(Path(cases) / "cases.json")
    assert result["noise_generator"] == "numpy" and result["recursion"] == "runtime"
    for row in result["results"]:
        assert row["denoise_steps"] == 1 and row["recursion_levels"] == 3
        assert row["protected_max_error"] in (None, 0.0), (path, row["case"])
        assert all(np.isfinite(v) for v in row.values() if isinstance(v, (int, float)))
    return result


def values(result):
    return {(r["subset"], r["subnet"]): r for r in summaries(result)}


def paired(candidate, baseline, manifest):
    merged = dict(
        candidate,
        results=[dict(r, method="candidate") for r in candidate["results"]]
        + [dict(r, method="control") for r in baseline["results"]],
    )
    return contrast(merged, merged, manifest, "control")


def screen(root, candidate, baseline, protocol, *, step=10000):
    root = Path(root)
    cases = root / "development"
    result = dict(candidate=candidate, baseline=baseline, step=step, violations=[], modes={})
    for mode in ("raw", "ema"):
        left = report(root / "arms" / candidate, step, mode, cases)
        right = report(root / "arms" / baseline, step, mode, cases)
        a, b = values(left), values(right)
        for group in protocol["screening_subsets"]:
            for index in range(3):
                for metric, tolerance in protocol["safety_tolerances"].items():
                    x, y = a[group, index].get(metric), b[group, index].get(metric)
                    if x is None and y is None:
                        continue
                    assert x is not None and y is not None
                    regression = y - x if metric == "active_ink_f1" else x - y
                    if regression > tolerance:
                        result["violations"].append(
                            dict(
                                mode=mode,
                                subset=group,
                                subnet=index,
                                metric=metric,
                                regression=regression,
                                tolerance=tolerance,
                            )
                        )
        result["modes"][mode] = dict(
            candidate_summary=summaries(left),
            baseline_summary=summaries(right),
            paired=paired(left, right, read(cases / "cases.json")),
        )
    improvements = []
    for row in result["modes"]["ema"]["paired"]:
        if row["subnet"] != 0 or row["subset"] != "sketch_geometry_strokes_strong":
            continue
        threshold = protocol["screening_improvement"].get(row["metric"])
        if threshold is None:
            continue
        lo, hi = row["clustered_95_interval"]
        improved = (
            row["delta"] >= threshold and lo > 0
            if row["metric"] == "active_ink_f1"
            else row["delta"] <= -threshold and hi < 0
        )
        if improved:
            improvements.append(row)
    result.update(improvements=improvements, adopt=not result["violations"] and bool(improvements))
    return result


def needs_receptive_field(root, arm, protocol):
    v = values(report(Path(root) / "arms" / arm, 10000, "ema", Path(root) / "development"))
    row = v["sketch_geometry_strokes_strong", 0]
    measured = {key: row[key] for key in protocol["a3_trigger"]}
    return dict(
        arm=arm,
        values=measured,
        thresholds=protocol["a3_trigger"],
        required=any(measured[k] > t for k, t in protocol["a3_trigger"].items()),
    )


def needs_continuation(root, arms, before, after, protocol):
    reasons = []
    for arm in arms:
        a = values(report(Path(root) / "arms" / arm, before, "ema", Path(root) / "development"))[
            "sketch_geometry_strokes_strong", 0
        ]
        b = values(report(Path(root) / "arms" / arm, after, "ema", Path(root) / "development"))[
            "sketch_geometry_strokes_strong", 0
        ]
        for key, threshold in protocol["continuation_improvement"].items():
            gain = b[key] - a[key] if key == "active_ink_f1" else a[key] - b[key]
            if gain >= threshold:
                reasons.append(dict(arm=arm, metric=key, improvement=gain, threshold=threshold))
    return dict(before=before, after=after, extend=bool(reasons), reasons=reasons)
