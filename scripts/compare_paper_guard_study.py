"""Compare paired paper-loss ablations with declared completed baseline trajectories."""

import argparse
from pathlib import Path

import numpy as np
from assess_ink_structure_study import assess
from campaign_protocol import read, sha, write
from compare_output_gradient_study import align_rows
from PIL import Image, ImageDraw
from summarize_lineart_loss_study import audit, differences, summarize
from train_lineart_loss_study import verify

from lineartgen.config import load_config
from lineartgen.training.lineart_study import study_methods


def settings(path, worktree):
    config = load_config(path).model_dump(mode="json")
    for key in ("root", "prepared_root"):
        if config["data"][key] is not None:
            config["data"][key] = str((worktree / config["data"][key]).resolve())
    return config


def paired_config(candidate, base, changes):
    expected = {
        key: value.copy() if isinstance(value, dict) else value for key, value in base.items()
    }
    for section, overrides in changes.items():
        expected[section].update(overrides)
    if candidate != expected:
        raise ValueError("Training settings differ beyond declared intervention")


def pairing(root, step):
    protocol = verify(root)
    assert sha(root / "plan.json") == protocol["plan_sha256"]
    assert read(root / "baseline-replay-audit.json")["full_model_optimizer_ema_exact"]
    all_parents = {
        x["root"]: x for entries in protocol["external_baselines"].values() for x in entries
    }
    for directory, declaration in all_parents.items():
        parent = Path(directory)
        assert sha(parent / "source.json") == declaration["source_sha256"]
        source = read(parent / "source.json")
        for path, digest in source["files"].items():
            assert sha(parent.parent.parent / path) == digest, path
            if path.startswith("src/lineartgen/data/") or path in (
                "src/lineartgen/diffusion.py",
                "src/lineartgen/sketch.py",
                "src/lineartgen/training/steps.py",
                "src/lineartgen/training/lineart_steps.py",
                "src/lineartgen/training/lineart_study.py",
                "scripts/evaluate_lineart_loss_study.py",
            ):
                assert sha(Path(path)) == digest, path
        for path, digest in source["settings"].items():
            assert sha(parent / path) == digest, path
        for name in (
            "evaluation.json",
            "dataset.json",
            "acceptance.json",
            "confirmation-plan.json",
        ):
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
    for name in study_methods(protocol):
        declaration = protocol["config_counterparts"][name]
        parent = Path(declaration["root"])
        assert str(parent) in all_parents
        paired_config(
            settings(root / f"configs/{name}.json", Path.cwd()),
            settings(parent / f"configs/{declaration['method']}.json", parent.parent.parent),
            declaration["changes"],
        )
    checkpoint = read(root / "checkpoint.json")
    for external in protocol["external_baselines"][str(step)]:
        before = read(Path(external["root"]) / "checkpoint.json")
        assert before["step"] == checkpoint["step"] == step
        assert before["clean_sha256"] == checkpoint["clean_sha256"]
    return protocol


def combine_panels(sources, output, selected, *, initial=False):
    """sources: iterable of (existing image, method labels); preserve every pixel."""
    prefix = ["target", "source", "noisy"] + (["initial"] if initial else [])
    columns, input_pixels, size = {}, None, None
    hashes = {}
    for path, methods in sources:
        with Image.open(path) as image:
            labels = prefix + list(methods)
            n = image.height - 24
            assert image.mode == "L" and image.width == n * len(labels)
            current = np.asarray(image)[24:, : n * len(prefix)]
            if input_pixels is None:
                input_pixels, size = current.copy(), n
                columns.update(
                    {
                        label: image.crop((i * n, 24, (i + 1) * n, n + 24))
                        for i, label in enumerate(prefix)
                    }
                )
            else:
                assert n == size
                np.testing.assert_array_equal(current, input_pixels)
            for label in selected:
                if label in methods:
                    if label in columns:
                        raise ValueError("Duplicate model column")
                    i = labels.index(label)
                    columns[label] = image.crop((i * n, 24, (i + 1) * n, n + 24))
        hashes[str(path)] = sha(path)
    labels = prefix + list(selected)
    canvas = Image.new("L", (size * len(labels), size + 24), 255)
    for i, label in enumerate(labels):
        canvas.paste(columns[label], (i * size, 24))
        ImageDraw.Draw(canvas).text((i * size + 5, 5), label, fill=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return hashes | {str(output): sha(output)}


def compare(root, step):
    protocol = pairing(root, step)
    methods = study_methods(protocol)
    declarations = protocol["external_baselines"][str(step)]
    reference = read(root / "evaluation.json")
    manifest_path = Path(reference["root"]) / "cases.json"
    assert sha(manifest_path) == reference["cases_sha256"]
    manifest = read(manifest_path)
    for file, digest in reference["files"].items():
        assert sha(Path(reference["root"]) / file) == digest
    initial, artifacts = audit(root, 0, "ema", reference, len(manifest["cases"]))
    for declaration in declarations:
        before, files = audit(
            Path(declaration["root"]), 0, "ema", reference, len(manifest["cases"])
        )
        assert before["results"] == initial["results"], "Initial inference differs"
        artifacts.update(files)
    milestones = {}
    basenames = [name for d in declarations for name in d["methods"]]
    assert len(set(basenames + list(methods))) == len(basenames) + len(methods)
    gates = {}
    for mode in ("ema", "raw"):
        own, files = audit(root, step, mode, reference, len(manifest["cases"]))
        artifacts.update(files)
        rows = own["results"].copy()
        for declaration in declarations:
            before, files = audit(
                Path(declaration["root"]), step, mode, reference, len(manifest["cases"])
            )
            artifacts.update(files)
            baseline = [r for r in before["results"] if r["method"] in declaration["methods"]]
            for name in declaration["methods"]:
                align_rows(own["results"], [r for r in baseline if r["method"] == name], methods)
            rows.extend(baseline)
        summary = summarize(rows)
        milestones[f"{step}-{mode}"] = dict(
            summary=summary,
            paired=[r for name in basenames for r in differences(rows, rows, manifest, name)]
            + differences(rows, initial["results"], manifest, "initial"),
        )
        rules = read(root / "acceptance.json")
        gates[mode] = {
            phase: [assess(summary, rules, n, screening=screening) for n in methods]
            for phase, screening in [("screen", True), ("final", False)]
        }
    analysis = root / f"analysis-with-baselines-{step}.json"
    write(
        analysis,
        dict(
            through=step,
            initial=dict(summary=summarize(initial["results"])),
            milestones=milestones,
            artifacts=artifacts,
            analysis_source_sha256=sha(Path(__file__)),
            limitations="One fine-tuning seed, synthetic development. "
            "Baselines reused, not refitted. "
            "Causal comparisons require named counterparts; fresh confirmation remains.",
        ),
    )
    results = gates["ema"]["screen" if step == 2000 else "final"]
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
            diagnostics=gates,
            analysis_sha256=sha(analysis),
            rules_sha256=sha(root / "acceptance.json"),
            meaning="Numerical eligibility only. Visual review and independent "
            "confirmations required.",
        ),
    )
    panels = {}
    selected = ["control"] + (["ste_ink_guard"] if "ste_ink_guard" in basenames else ["ink_guard"])
    selected += list(methods)
    for suite in ("evaluation", "supplemental-evaluation"):
        for mode in ("ema", "raw"):
            for suffix in ("", "-diagnostic") if suite == "evaluation" else ("",):
                relative = Path(suite) / f"step-{step}-{mode}{suffix}"
                for path in sorted((root / relative).glob("*-krita-white.png")):
                    sources = [(path, methods)]
                    for declaration in declarations:
                        parent = Path(declaration["root"])
                        if any(n in selected for n in declaration["methods"]):
                            sources.append(
                                (
                                    parent / relative / path.name,
                                    study_methods(read(parent / "protocol.json")),
                                )
                            )
                    if not all(p.exists() for p, _ in sources):
                        continue
                    panels.update(
                        combine_panels(
                            sources,
                            root / "comparison-panels" / relative / path.name,
                            selected,
                            initial=suite == "supplemental-evaluation",
                        )
                    )
    write(
        root / f"comparison-panels-{step}.json",
        dict(files=panels, labels=selected, script_sha256=sha(Path(__file__))),
    )
    print([(r["method"], r["numerical_gate_passed"], len(r["failures"])) for r in results])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    compare(args.root.resolve(), args.step)
