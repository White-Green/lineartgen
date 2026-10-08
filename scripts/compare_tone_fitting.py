"""Apply the predeclared tone-weight local screen without selecting other steps."""

import argparse
from pathlib import Path

from campaign_protocol import read, sha, write
from diagnose_learnability import verify
from evaluate_conditional_fitting import check_rows


def compare(root):
    verify(root)
    plan = read(root / "plan.json")
    original, final = root / "evaluation/reference-1200", root / "evaluation/step-1200"
    assert read(original / "historical-one-step-audit.json")["all_exact"]
    assert read(root / "baseline-replay-audit.json")["full_state_exact"]
    restore = read(root / "training/restore-audit.json")
    assert restore["step"] == 1200 and restore["all_state_exact"]
    for folder, names in ((original, ["warm_detail16"]), (final, plan["arms"])):
        report = read(folder / "metrics.json")
        check_rows(report["results"], names, (False, True), (1, 2, 5))
        for path, digest in report["artifacts"].items():
            assert sha(path) == digest, path
    summaries = read(final / "summary.json") + read(original / "summary.json")
    index = {
        (r["method"], r["split"], r["family"], r["partial"], r["strength"]): r
        for r in summaries
        if r["mode"] == "raw" and r["denoise_steps"] == 1
    }
    results = []
    for method in plan["arms"]:
        failures, comparisons, gains = [], [], {}
        for split in ("train", "probe"):
            for family in ("clean", "tone", "blur"):
                for partial in (False, True):
                    for strength in (0.1, 0.5):
                        key = split, family, partial, strength
                        before, after = index[("warm_detail16", *key)], index[(method, *key)]
                        delta = {k: after["means"][k] - before["means"][k] for k in before["means"]}
                        item = dict(
                            split=split,
                            family=family,
                            partial=partial,
                            strength=strength,
                            delta=delta,
                            absolute_fit_failures=len(after["failed_fit"]),
                        )
                        reasons = []
                        if family == "blur" and strength == 0.1 and not partial:
                            gain = (
                                1
                                - after["means"]["halo_opacity_excess"]
                                / before["means"]["halo_opacity_excess"]
                            )
                            gains[split] = gain
                            if gain < 0.1:
                                reasons.append("low-strength blur halo reduction below10%")
                        if delta["ink_f1_alpha50"] < -0.005:
                            reasons.append("F1 drops over .005")
                        if delta["ink_deficit_fraction"] > 0.005:
                            reasons.append("ink deficit increases over .005")
                        if delta["halo_opacity_excess"] > 0.01:
                            reasons.append("halo increases over .01")
                        comparisons.append(item)
                        if reasons:
                            failures.append(dict(item, reasons=reasons))
        results.append(
            dict(
                method=method,
                numerical_probe_pass=not failures,
                primary_halo_gains=gains,
                failures=failures,
                comparisons=comparisons,
            )
        )
    write(
        root / "matched-results.json",
        dict(
            results=results,
            primary="raw",
            primary_steps=1,
            adoption=False,
            visual_review_required=True,
            source_sha256=sha(root / "source.json"),
            reference_metrics_sha256=sha(original / "metrics.json"),
            candidate_metrics_sha256=sha(final / "metrics.json"),
            reference_summary_sha256=sha(original / "summary.json"),
            candidate_summary_sha256=sha(final / "summary.json"),
        ),
    )
    print(
        [
            {k: r[k] for k in ("method", "numerical_probe_pass", "primary_halo_gains")}
            for r in results
        ],
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    compare(p.parse_args().root.resolve())
