"""Use unchanged tone-study gates and record the fixed tempered-reference deltas."""

import argparse
from pathlib import Path

from campaign_protocol import read, sha, write
from compare_tone_fitting import compare
from prepare_opacity_fitting import verify_tempered


def compare_opacity(root):
    compare(root)  # The predeclared balanced-reference primary gate is unchanged.
    reference = verify_tempered(root) / "evaluation/step-1200"
    fields = ("mode", "split", "denoise_steps", "family", "partial", "strength")
    old = {
        tuple(r[k] for k in fields): r
        for r in read(reference / "summary.json")
        if r["method"] == "warm_tone_sqrt"
    }
    rows = []
    for r in read(root / "evaluation/step-1200/summary.json"):
        before = old[tuple(r[k] for k in fields)]
        assert r["conditions"] == before["conditions"]
        rows.append(
            dict(
                method=r["method"],
                **{k: r[k] for k in fields},
                before=before["means"],
                after=r["means"],
                delta={k: r["means"][k] - before["means"][k] for k in before["means"]},
                absolute_fit_failures=len(r["failed_fit"]),
            )
        )
    assert len(rows) == 432
    write(
        root / "tempered-reference-deltas.json",
        dict(
            comparisons=rows,
            descriptive_only=True,
            adoption=False,
            primary_gate="matched-results.json",
            reference_summary_sha256=sha(reference / "summary.json"),
            plan_sha256=sha(root / "plan.json"),
        ),
    )
    print(dict(tempered_comparisons=len(rows), primary_gate_unchanged=True), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    compare_opacity(p.parse_args().root.resolve())
