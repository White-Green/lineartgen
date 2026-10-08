"""Compare the broad detail branch with its declared guided and campaign controls."""

import argparse
from pathlib import Path

from assess_ink_structure_study import assess
from campaign_protocol import read, sha, write
from compare_output_gradient_study import align_rows
from compare_source_guide_study import combine_panels, paired_config, settings
from parameter_blend_study import original_source
from summarize_lineart_loss_study import audit, differences, summarize
from train_lineart_loss_study import verify

from lineartgen.training.lineart_study import study_methods


def compare(root):
    protocol = verify(root)
    plan = read(root / "plan.json")
    assert sha(root / "plan.json") == protocol["plan_sha256"]
    assert read(root / "baseline-replay-audit.json")["full_model_optimizer_ema_exact"]
    methods = study_methods(protocol)
    assert len(methods) == 1
    pair = protocol["counterpart"]
    parent, baseline = Path(pair["root"]), pair["method"]
    control = Path(plan["control"]["root"])
    assert sha(control / "source.json") == plan["control"]["source_sha256"]
    declared = next(x for x in plan["parents"].values() if x["root"] == str(parent))
    assert declared["method"] == baseline
    assert sha(parent / "source.json") == declared["source_sha256"]
    assert read(root / "checkpoint.json")["step"] == 2000
    assert (
        read(root / "checkpoint.json")["clean_sha256"]
        == read(parent / "checkpoint.json")["clean_sha256"]
    )
    paired_config(
        settings(root / f"configs/{methods[0]}.json", Path.cwd()),
        settings(parent / f"configs/{baseline}.json", parent.parent.parent),
        {"model": {"detail_channels": 16}},
    )
    for folder in (parent, control):
        original_source(folder)
        for name in (
            "dataset.json",
            "evaluation.json",
            "acceptance.json",
            "confirmation-plan.json",
        ):
            assert read(root / name) == read(folder / name)
        original = read(folder / "protocol.json")
        for key in (
            "initial",
            "initial_files",
            "total_schedule_steps",
            "patch_size",
            "selected_panels",
        ):
            assert protocol[key] == original[key]
    reference = read(root / "evaluation.json")
    manifest_path = Path(reference["root"]) / "cases.json"
    assert sha(manifest_path) == reference["cases_sha256"]
    manifest = read(manifest_path)
    for name, digest in reference["files"].items():
        assert sha(Path(reference["root"]) / name) == digest
    initial, artifacts = audit(root, 0, "ema", reference, len(manifest["cases"]))
    for folder in (parent, control):
        old, files = audit(folder, 0, "ema", reference, len(manifest["cases"]))
        assert old["results"] == initial["results"]
        artifacts.update(files)
    modes = {}
    gates = {}
    for mode in ("ema", "raw"):
        own, files = audit(root, 2000, mode, reference, len(manifest["cases"]))
        artifacts.update(files)
        paired, files = audit(parent, 2000, mode, reference, len(manifest["cases"]))
        artifacts.update(files)
        fixed, files = audit(control, 2000, mode, reference, len(manifest["cases"]))
        artifacts.update(files)
        before = [x for x in paired["results"] if x["method"] == baseline]
        reference_rows = [x for x in fixed["results"] if x["method"] == "control"]
        for rows in (before, reference_rows):
            align_rows(own["results"], rows, methods)
        summary = summarize(own["results"] + before + reference_rows)
        modes[mode] = dict(
            summary=summary,
            paired=differences(own["results"], before, manifest, baseline)
            + differences(own["results"], reference_rows, manifest, "control")
            + differences(own["results"], initial["results"], manifest, "initial"),
        )
        rules = read(root / "acceptance.json")
        gates[mode] = {
            phase: [assess(summary, rules, n, screening=screen) for n in methods]
            for phase, screen in [("screen", True), ("final", False)]
        }
    write(
        root / "analysis-with-baselines-2000.json",
        dict(
            step=2000,
            modes=modes,
            artifacts=artifacts,
            source_sha256=sha(Path(__file__)),
            limitations=(
                "One training seed; previously observed synthetic development; "
                "visual review and independent confirmation remain."
            ),
        ),
    )
    results = gates["ema"]["screen"]
    write(
        root / "quality-gate-with-baselines-2000.json",
        dict(
            step=2000,
            results=results,
            diagnostics=gates,
            ranked_provisional_candidates=[
                r["method"] for r in results if r["numerical_gate_passed"]
            ],
            rules_sha256=sha(root / "acceptance.json"),
            adoption=False,
        ),
    )
    selected = ["control", baseline, *methods]
    panels = {}
    for suite in ("evaluation", "supplemental-evaluation"):
        for mode in ("ema", "raw"):
            for suffix in ("", "-diagnostic") if suite == "evaluation" else ("",):
                relative = Path(suite) / f"step-2000-{mode}{suffix}"
                for path in sorted((root / relative).glob("*-krita-white.png")):
                    sources = [(path, methods)] + [
                        (p / relative / path.name, study_methods(read(p / "protocol.json")))
                        for p in (control, parent)
                    ]
                    if all(p.exists() for p, _ in sources):
                        panels.update(
                            combine_panels(
                                sources,
                                root / "comparison-panels" / relative / path.name,
                                selected,
                                initial=suite == "supplemental-evaluation",
                            )
                        )
    write(root / "comparison-panels-2000.json", dict(files=panels, labels=selected))
    print([(x["method"], x["numerical_gate_passed"], len(x["failures"])) for x in results])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    compare(p.parse_args().root.resolve())
