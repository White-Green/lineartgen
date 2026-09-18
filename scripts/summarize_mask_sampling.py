"""Separate N2's full/partial editing and strength slices on the frozen suite."""

import argparse
import json
from pathlib import Path

from run_sketch_rates import validate
from summarize_sketch_rates import contrast, summaries
from summarize_training_methods import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--through", type=int, choices=[2000, 10000], default=10000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.root / "evaluation/cases.json").read_text())
    files = [
        args.root / "evaluation/cases.json",
        args.root / "reference-reuse.json",
        args.root / "sampling-diagnostic.json",
        args.root / "sampling-training-replay.json",
    ]
    reports = {}
    for step in (2000,) if args.through == 2000 else (2000, 10000):
        for mode in ("raw", "ema"):
            path = args.root / f"evaluation/step-{step}-{mode}/metrics.json"
            report = json.loads(path.read_text())
            validate(report, ("control", "boundary8"), step, "runtime")
            assert report["manifest_sha256"] == sha(args.root / "evaluation/cases.json")
            lookup = {
                (r["case"], r["subnet"]): r for r in report["results"] if r["method"] == "control"
            }
            for row in report["results"]:
                ref = lookup[row["case"], row["subnet"]]
                for field in ("edit_boundary_pixels", "edit_boundary_ink_pixels"):
                    assert row[field] == ref[field]
            files.append(path)
            slices = {}
            for partial in (False, True):
                for strength in (None, 0.1, 0.5):
                    selected = dict(
                        report,
                        results=[
                            r
                            for r in report["results"]
                            if r["partial"] == partial
                            and (strength is None or r["strength"] == strength)
                        ],
                    )
                    scope = "partial" if partial else "full"
                    label = f"{scope}-{strength if strength is not None else 'all'}"
                    slices[label] = dict(
                        summary=summaries(selected),
                        paired_minus_control=contrast(selected, selected, manifest, "control"),
                    )
            reports[f"{step}-{mode}"] = slices
    args.output.write_text(
        json.dumps(
            dict(
                through=args.through,
                slices=reports,
                artifacts={str(p): sha(p) for p in files},
                note="Boundary scores cover editable pixels within 3 pixels of protected pixels. "
                "Ink F1/MAE require target ink. Intervals condition on one fitted seed.",
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    print(args.output)


if __name__ == "__main__":
    main()
