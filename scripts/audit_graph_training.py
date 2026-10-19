"""Audit the completed continuation from its actual update and sampling records."""

import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
from graph_long_training import check


def audit(root):
    check(root)
    plan = json.loads((root / "plan.json").read_text())
    summary = json.loads((root / "training-summary.json").read_text())
    rows = [json.loads(s) for s in (root / "training/metrics.jsonl").read_text().splitlines()]
    samples = [json.loads(s) for s in (root / "training/sampling.jsonl").read_text().splitlines()]
    count = plan["added_steps"]
    assert len(rows) == len(samples) == summary["added_steps"] == count
    assert [r["step"] for r in rows] == list(range(1, count + 1))
    assert [r["step"] for r in samples] == list(range(1, count + 1))
    assert all(r["total_steps"] == plan["parent_steps"] + r["step"] for r in rows)
    assert all(np.isfinite(list(r.values())).all() for r in rows)
    depths = dict(collections.Counter(r["depth"] for r in rows))
    assert depths == dict.fromkeys(range(1, 7), count // 6)
    assert all(
        r["depth"] == s["depth"] == (r["step"] - 1) % 6 + 1
        for r, s in zip(rows, samples, strict=True)
    )
    assert abs(max(r["learning_rate"] for r in rows) - 2e-4) < 1e-9
    assert abs(rows[-1]["learning_rate"] - 2e-6) < 1e-9
    weights = {}
    for mode in ("raw", "ema"):
        path = root / f"snapshots/step-{count}/{mode}.npz"
        with np.load(path, allow_pickle=False) as z:
            assert sum(a.size for a in z.values()) == 22725
            assert all(np.isfinite(a).all() for a in z.values())
        weights[mode] = dict(
            path=str(path.relative_to(root)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )
    windows = []
    for start in range(0, count, 6000):
        window = rows[start : start + 6000]
        windows.append(
            dict(
                start=start + 1,
                end=window[-1]["step"],
                by_depth={
                    d: {
                        k: float(np.mean([r[k] for r in window if r["depth"] == d]))
                        for k in ("loss", "vertex", "edge", "position", "width", "fill")
                    }
                    for d in range(1, 7)
                },
            )
        )
    report = dict(
        passed=True,
        parent_steps=plan["parent_steps"],
        added_steps=count,
        cumulative_steps=plan["parent_steps"] + count,
        depth_counts=depths,
        finite=True,
        weights=weights,
        peak_learning_rate=max(r["learning_rate"] for r in rows),
        final_learning_rate=rows[-1]["learning_rate"],
        clipping_fraction=float(np.mean([r["gradient_norm"] > 1 for r in rows])),
        partial_fraction=float(np.mean([p for r in samples for p in r["partial"]])),
        extra_fraction=float(np.mean([v for r in samples for v in r["replace"]["control"]])),
        windows=windows,
        note=(
            "Update counts and finite training are verified; "
            "this alone does not establish visual quality or convergence."
        ),
    )
    (root / "training-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "windows"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    audit(parser.parse_args().root)
