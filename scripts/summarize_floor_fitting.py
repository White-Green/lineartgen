"""Matched controlled-fit comparisons; a pass only proposes a broader experiment."""

import argparse
from pathlib import Path

import numpy as np
from campaign_protocol import read, sha, write
from diagnose_learnability import verify


def summarize(root):
    verify(root)
    plan = read(root / "plan.json")
    parent = Path(plan["parent"])
    verify(parent)
    assert read(root / "training/restore-audit.json")["step"] == 1200
    assert read(parent / "training/restore-audit.json")["step"] == 1200
    baseline = read(parent / "evaluation/step-1200/summary.json")
    candidate = read(root / "evaluation/step-1200/summary.json")
    initial_checks = []
    results = []
    for name, counterpart in plan["counterparts"].items():
        for mode in ("raw", "ema"):
            for split in ("train", "probe"):
                before = parent / f"evaluation/step-0/{counterpart}-{mode}-{split}.npz"
                after = root / f"evaluation/step-0/{name}-{mode}-{split}.npz"
                np.testing.assert_array_equal(np.load(before)["images"], np.load(after)["images"])
                initial_checks.append(dict(method=name, mode=mode, split=split, exact=True))
        comparisons, failures = [], []
        for split in ("train", "probe"):
            for family in ("clean", "tone", "blur"):

                def row(rows, method, split=split, family=family):
                    return next(
                        x
                        for x in rows
                        if (x["method"], x["mode"], x["split"], x["family"])
                        == (method, "raw", split, family)
                    )

                old, new = row(baseline, counterpart), row(candidate, name)
                delta = {k: new["means"][k] - old["means"][k] for k in old["means"]}
                comparisons.append(
                    dict(
                        split=split,
                        family=family,
                        delta=delta,
                        failed_individual_fit=len(new["failed_fit"]),
                        conditions=new["conditions"],
                    )
                )
                if family == "blur" and delta["ink_f1_alpha50"] < 0.03:
                    failures.append(
                        dict(split=split, family=family, reason="blur F1 gain below .03")
                    )
                if family == "clean" and delta["ink_f1_alpha50"] < -0.01:
                    failures.append(
                        dict(split=split, family=family, reason="clean F1 drops over .01")
                    )
                if delta["halo_opacity_excess"] > 0.01:
                    failures.append(dict(split=split, family=family, reason="halo rises over .01"))
        results.append(
            dict(
                method=name,
                counterpart=counterpart,
                numerical_probe_pass=not failures,
                failures=failures,
                comparisons=comparisons,
            )
        )
    write(
        root / "matched-results.json",
        dict(
            results=results,
            initial_checks=initial_checks,
            primary="raw",
            adoption=False,
            visual_review_required=True,
            parent_metrics_sha256=sha(parent / "evaluation/step-1200/metrics.json"),
            candidate_metrics_sha256=sha(root / "evaluation/step-1200/metrics.json"),
            source_sha256=sha(root / "source.json"),
        ),
    )
    print([(x["method"], x["numerical_probe_pass"], len(x["failures"])) for x in results])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    summarize(p.parse_args().root.resolve())
