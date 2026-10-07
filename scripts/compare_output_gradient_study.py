"""Compare surrogate-gradient arms with the separately trained, paired parent baselines."""

import argparse
from pathlib import Path

import numpy as np
from assess_ink_structure_study import assess
from campaign_protocol import read, sha, write
from PIL import Image, ImageDraw
from summarize_lineart_loss_study import audit, differences, summarize
from train_lineart_loss_study import verify

from lineartgen.config import load_config
from lineartgen.training.lineart_study import study_methods

CASE_FIELDS = (
    "case",
    "domain",
    "kind",
    "source",
    "suite",
    "family",
    "severity",
    "strength",
    "partial",
    "denoise_steps",
)


def align_rows(rows, reference, methods):
    """Refuse missing, duplicated or differently conditioned paired observations."""
    baseline = {(r["case"], r["denoise_steps"]): r for r in reference}
    if len(baseline) != len(reference):
        raise ValueError("Duplicate baseline observations")
    expected = {(case, steps, method) for case, steps in baseline for method in methods}
    actual = [(r["case"], r["denoise_steps"], r["method"]) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Candidate observations do not cover the paired cases exactly")
    for row in rows:
        before = baseline[row["case"], row["denoise_steps"]]
        if any(row[key] != before[key] for key in CASE_FIELDS):
            raise ValueError(f"Mismatched case conditions: {row['case']}")


def pairing(root, step):
    protocol = verify(root)
    external = protocol["external_baselines"][str(step)]
    parent = Path(external["root"])
    assert external["methods"] == ["control", "ink_guard"]
    assert sha(parent / "source.json") == external["source_sha256"]
    parent_source = read(parent / "source.json")
    parent_worktree = parent.parent.parent
    for path, digest in parent_source["files"].items():
        assert sha(parent_worktree / path) == digest, path
    for path, digest in parent_source["settings"].items():
        assert sha(parent / path) == digest, path
    for path, digest in parent_source["files"].items():
        if path.startswith("src/lineartgen/data/") or path in (
            "src/lineartgen/diffusion.py",
            "src/lineartgen/sketch.py",
            "src/lineartgen/training/steps.py",
            "src/lineartgen/training/lineart_steps.py",
            "src/lineartgen/training/lineart_study.py",
            "src/lineartgen/training/losses.py",
            "scripts/evaluate_lineart_loss_study.py",
        ):
            assert sha(Path(path)) == digest, path
    for name in ("evaluation.json", "dataset.json", "acceptance.json", "confirmation-plan.json"):
        assert read(root / name) == read(parent / name), name
    original = read(parent / "protocol.json")
    for key in (
        "initial",
        "initial_files",
        "total_schedule_steps",
        "patch_size",
        "selected_panels",
    ):
        assert protocol[key] == original[key], key

    def settings(path, worktree):
        config = load_config(path).model_dump(mode="json")
        data_root = Path(config["data"]["root"])
        config["data"]["root"] = str(
            (data_root if data_root.is_absolute() else worktree / data_root).resolve()
        )
        return config

    for name in study_methods(protocol):
        counterpart = protocol["hard_counterparts"][name]
        base = settings(parent / f"configs/{counterpart}.json", parent_worktree)
        candidate = settings(root / f"configs/{name}.json", Path.cwd())
        assert candidate["model"]["output_clip_gradient"] == "straight_through"
        assert base["model"]["output_clip_gradient"] == "hard"
        candidate["model"]["output_clip_gradient"] = "hard"
        assert candidate == base, f"Unpaired training configuration: {name}"
    # Both trainers also record the actual final indexed clean batch, before noise.
    a, b = read(root / "checkpoint.json"), read(parent / "checkpoint.json")
    assert a["step"] == b["step"] == step
    assert a["clean_sha256"] == b["clean_sha256"]
    return protocol, parent


def combine_panel(source, baseline_source, output, methods, parent_methods, *, initial=False):
    """Combine existing uint8 outputs, asserting identical teacher/input columns."""
    prefix = ["target", "source", "noisy"] + (["initial"] if initial else [])
    labels, parent_labels = prefix + list(methods), prefix + list(parent_methods)
    selected = prefix + ["control", "ink_guard"] + list(methods)
    with Image.open(source) as current, Image.open(baseline_source) as baseline:
        size = current.height - 24
        assert current.mode == baseline.mode == "L" and current.height == baseline.height
        assert current.width == len(labels) * size and baseline.width == len(parent_labels) * size
        for index in range(len(prefix)):
            rectangle = (size * index, 24, size * (index + 1), size + 24)
            np.testing.assert_array_equal(
                np.asarray(current.crop(rectangle)), np.asarray(baseline.crop(rectangle))
            )
        canvas = Image.new("L", (size * len(selected), size + 24), 255)
        for index, name in enumerate(selected):
            image, columns = (
                (baseline, parent_labels) if name in ("control", "ink_guard") else (current, labels)
            )
            column = columns.index(name)
            canvas.paste(
                image.crop((column * size, 24, (column + 1) * size, size + 24)), (index * size, 24)
            )
            ImageDraw.Draw(canvas).text((index * size + 5, 5), name, fill=0)
        output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output)
    return {str(p): sha(p) for p in (source, baseline_source, output)}


def compare(root, step):
    protocol, parent = pairing(root, step)
    methods = study_methods(protocol)
    reference = read(root / "evaluation.json")
    manifest_path = Path(reference["root"]) / "cases.json"
    assert sha(manifest_path) == reference["cases_sha256"]
    manifest = read(manifest_path)
    for path, digest in reference["files"].items():
        assert sha(Path(reference["root"]) / path) == digest, path
    initial, artifacts = audit(parent, 0, "ema", reference, len(manifest["cases"]))
    own_initial, files = audit(root, 0, "ema", reference, len(manifest["cases"]))
    assert own_initial["results"] == initial["results"], "Initial inference differs"
    artifacts.update(files)
    milestones = {}
    for mode in ("ema", "raw"):
        own, files = audit(root, step, mode, reference, len(manifest["cases"]))
        artifacts.update(files)
        external, files = audit(parent, step, mode, reference, len(manifest["cases"]))
        artifacts.update(files)
        baselines = [r for r in external["results"] if r["method"] in ("control", "ink_guard")]
        for name in ("control", "ink_guard"):
            align_rows(own["results"], [r for r in baselines if r["method"] == name], methods)
        rows = baselines + own["results"]
        milestones[f"{step}-{mode}"] = dict(
            summary=summarize(rows),
            paired=[
                r
                for name in ("control", "ink_guard")
                for r in differences(rows, rows, manifest, name)
            ]
            + differences(rows, initial["results"], manifest, "initial"),
        )
    analysis = root / f"analysis-with-baselines-{step}.json"
    write(
        analysis,
        dict(
            through=step,
            initial=dict(summary=summarize(initial["results"])),
            milestones=milestones,
            artifacts=artifacts,
            external_baselines=protocol["external_baselines"],
            analysis_source_sha256=sha(Path(__file__)),
            limitations="Paired synthetic development only; one generator training seed. "
            "Baseline trajectories are reused from the parent study, not separately refitted. "
            "Visual review, fresh conditions and another seed remain required.",
        ),
    )
    rules = read(root / "acceptance.json")
    results = [
        assess(milestones[f"{step}-ema"]["summary"], rules, method, screening=step == 2000)
        for method in methods
    ]
    ranking = sorted(
        [r for r in results if r["numerical_gate_passed"]],
        key=lambda r: -sum(r["primary_halo_gains"].values()),
    )
    write(
        root / f"quality-gate-with-baselines-{step}.json",
        dict(
            step=step,
            results=results,
            ranked_provisional_candidates=[r["method"] for r in ranking],
            analysis_sha256=sha(analysis),
            rules_sha256=sha(root / "acceptance.json"),
            script_sha256=sha(Path(__file__)),
            meaning="Provisional numerical gate; manual visual and independent confirmations "
            "pending",
        ),
    )
    parent_methods = study_methods(read(parent / "protocol.json"))
    panels = {}
    for suite in ("evaluation", "supplemental-evaluation"):
        for mode in ("ema", "raw"):
            for suffix in ("", "-diagnostic") if suite == "evaluation" else ("",):
                relative = Path(suite) / f"step-{step}-{mode}{suffix}"
                for source in sorted((root / relative).glob("*-krita-white.png")):
                    baseline_source = parent / relative / source.name
                    if not baseline_source.exists():
                        continue
                    output = root / "comparison-panels" / relative / source.name
                    panels.update(
                        combine_panel(
                            source,
                            baseline_source,
                            output,
                            methods,
                            parent_methods,
                            initial=suite == "supplemental-evaluation",
                        )
                    )
    write(
        root / f"comparison-panels-{step}.json",
        dict(files=panels, script_sha256=sha(Path(__file__))),
    )
    print([(r["method"], r["numerical_gate_passed"], len(r["failures"])) for r in results])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    compare(args.root.resolve(), args.step)
