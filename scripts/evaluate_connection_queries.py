"""Independent widest-path checks and predeclared paired fitting gates."""

import argparse
import heapq
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from scipy import ndimage
from train_connection_queries import CASES, METHODS, STEPS, query_arrays

from lineartgen.evaluation import _connected
from lineartgen.training.connectivity import connection_capacity


def widest(alpha, source, sink):
    best = np.full(alpha.shape, -1.0)
    queue = []
    for y, x in np.argwhere(source):
        best[y, x] = alpha[y, x]
        heapq.heappush(queue, (-float(alpha[y, x]), int(y), int(x)))
    while queue:
        negative, y, x = heapq.heappop(queue)
        if -negative < best[y, x]:
            continue
        if sink[y, x]:
            return -negative
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = y + dy, x + dx
                if 0 <= yy < alpha.shape[0] and 0 <= xx < alpha.shape[1]:
                    candidate = min(-negative, float(alpha[yy, xx]))
                    if candidate > best[yy, xx]:
                        best[yy, xx] = candidate
                        heapq.heappush(queue, (-candidate, yy, xx))
    return 0.0


def whole_capacity(alpha, query):
    lo, hi = 0.0, 1.0
    for _ in range(16):
        mid = (lo + hi) / 2
        labels = ndimage.label(alpha > mid, np.ones((3, 3)))[0]
        if _connected(labels, query):
            lo = mid
        else:
            hi = mid
    return lo


def audit(root):
    cache = np.load(root / "inputs.npz")
    initial = np.load(root / "evaluation/step-0/predictions.npz")
    run = jax.jit(jax.vmap(jax.value_and_grad(connection_capacity)))
    rows = []
    for split in ("train", "probe"):
        indices, a, b = query_arrays(root, split)
        a, b = np.asarray(a), np.asarray(b)
        ix = tuple(np.asarray(x) for x in indices)
        for mode in ("teacher", "initial"):
            images = np.stack(
                [
                    cache[f"{split}-{case}-targets-0"][0]
                    if mode == "teacher"
                    else initial[f"pixel-{split}-{case}-replicate_0"]
                    for case in CASES
                ]
            )
            patches = np.clip((1 - images[..., 0]) / 2, 0, 1)[ix]
            values, gradient = run(patches, a, b)
            for i, (p, sa, sb) in enumerate(zip(patches, a, b, strict=True)):
                expected = widest(p, sa, sb)
                assert float(values[i]) == expected
                assert np.isfinite(gradient[i]).all()
                assert np.count_nonzero(gradient[i]) == 1
                assert float(gradient[i].sum()) == 1
                rows.append(
                    dict(
                        split=split,
                        mode=mode,
                        query=i,
                        capacity=expected,
                        selected_gradient_finite=True,
                        exact_independent_oracle=True,
                    )
                )
    write(
        root / "query-oracle-audit.json",
        dict(
            rows=rows,
            all_exact=True,
            input_sha256=sha(root / "inputs.npz"),
            script_sha256=sha(Path(__file__)),
        ),
    )
    print("actual64px query oracle and gradients passed", len(rows), flush=True)


def evaluate(root, step):
    folder = root / f"evaluation/step-{step}"
    scores = read(folder / "metrics.json")["rows"]
    predictions = np.load(folder / "predictions.npz")
    cache = np.load(root / "inputs.npz")
    rows = []
    for row in scores:
        if row["case"] != "fit_gap":
            continue
        query = cache[f"{row['split']}-fit_gap-connections"][-1]
        key = "-".join(row[k] for k in ("method", "split", "case", "draw"))
        alpha = np.clip((1 - predictions[key][..., 0]) / 2, 0, 1)
        rows.append(
            dict(
                method=row["method"],
                split=row["split"],
                draw=row["draw"],
                whole_gap_capacity=whole_capacity(alpha, query),
            )
        )
    summaries, failures = [], []
    metrics = (
        "ink_f1_alpha50",
        "ink_deficit_fraction",
        "halo_opacity_excess",
        "profile_width_mae_px",
        "connection_cut_rate",
        "gap_middle_alpha",
    )
    for split in ("train", "probe"):
        for case in CASES:
            means = {}
            for method in METHODS:
                selected = [
                    r
                    for r in scores
                    if (r["split"], r["case"], r["method"]) == (split, case, method)
                ]
                assert len(selected) == 8
                means[method] = {
                    metric: float(np.mean([r[metric] for r in selected]))
                    for metric in metrics
                    if selected[0][metric] is not None
                }
                means[method]["passes"] = sum(r["fit_screen_pass"] for r in selected)
                if case == "fit_gap":
                    means[method]["whole_gap_capacity"] = float(
                        np.mean(
                            [
                                r["whole_gap_capacity"]
                                for r in rows
                                if r["method"] == method and r["split"] == split
                            ]
                        )
                    )
            baseline, treatment = means["pixel"], means["connection"]
            checks = dict(
                f1=treatment["ink_f1_alpha50"] >= baseline["ink_f1_alpha50"] - 0.005,
                ink=treatment["ink_deficit_fraction"] <= baseline["ink_deficit_fraction"] + 0.005,
                width=treatment["profile_width_mae_px"] <= baseline["profile_width_mae_px"] + 0.10,
                cuts=treatment["connection_cut_rate"] <= baseline["connection_cut_rate"],
            )
            if case == "fit_gap":
                checks["gap_reduction"] = (
                    treatment["gap_middle_alpha"] <= 0.75 * baseline["gap_middle_alpha"]
                )
            failures.extend(
                dict(split=split, case=case, check=k) for k, v in checks.items() if not v
            )
            summaries.append(dict(split=split, case=case, means=means, checks=checks))
    write(
        folder / "paired-gate.json",
        dict(
            step=step,
            summaries=summaries,
            failures=failures,
            numeric_advancement_pass=not failures,
            whole_gap_rows=rows,
            metrics_sha256=sha(folder / "metrics.json"),
            predictions_sha256=sha(folder / "predictions.npz"),
            plan_sha256=sha(root / "plan.json"),
            script_sha256=sha(Path(__file__)),
            diagnostic_only=True,
            adoption=False,
        ),
    )
    print(dict(step=step, paired_failures=len(failures)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "evaluate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--step", type=int, default=STEPS)
    args = parser.parse_args()
    if args.action == "audit":
        audit(args.root.resolve())
    else:
        evaluate(args.root.resolve(), args.step)
