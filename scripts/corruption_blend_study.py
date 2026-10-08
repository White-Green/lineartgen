"""Fixed small blends of guided strokes/mixed checkpoints, with strict final gates at 2k."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
from assess_ink_structure_study import assess
from campaign_protocol import read, sha, write
from compare_corruption_mixture_study import combine_panels, paired_config, settings
from compare_output_gradient_study import align_rows
from evaluate_lineart_loss_study import evaluate
from flax import nnx
from parameter_blend_study import freeze, original_source, verify_endpoints
from summarize_lineart_loss_study import audit, differences, summarize

from lineartgen.config import load_config, save_config
from lineartgen.evaluation_sampling import inference_with_source, runtime_inputs
from lineartgen.models.blending import blend_parameters, compatible_config
from lineartgen.models.serialization import load_inference, save_inference

ALPHAS = {"corruption_blend0625": 1 / 16, "corruption_blend125": 1 / 8, "corruption_blend25": 1 / 4}


def plan(path, strokes, mixed, control):
    if path.exists():
        raise FileExistsError(path)
    endpoints = {}
    for tag, parent, method in (
        ("strokes", strokes, "source_ink_guard"),
        ("mixed", mixed, "mixed_source_ink_guard"),
    ):
        original_source(parent)
        endpoints[tag] = dict(root=str(parent), method=method, modes={})
        for mode in ("ema", "raw"):
            folder = parent / f"snapshots/step-2000/{method}/{mode}"
            endpoints[tag]["modes"][mode] = {
                n: sha(folder / n) for n in ("weights.npz", "config.json", "metadata.json")
            }
    original_source(control)
    write(
        path,
        dict(
            alpha_map=ALPHAS,
            step=2000,
            endpoints=endpoints,
            control=str(control),
            acceptance_sha256=sha(control / "acceptance.json"),
            confirmation_plan_sha256=sha(control / "confirmation-plan.json"),
            decision=(
                "Strict final criteria at the fixed 2k checkpoints; "
                "never use the permissive screen for eligibility"
            ),
            ranking=(
                "Sum of primary EMA halo gains among all strict-final passing, "
                "visually acceptable models; select one"
            ),
            training=(
                "No additional training or optimizer states. "
                "Construct one 126828-parameter model, not averaged images"
            ),
            confirmation=(
                "Retrain both endpoints to exactly 2k with seed20261006, "
                "use the same selected alpha; then sealed fresh conditions"
            ),
            followup=(
                "No automatic 10k endpoint continuation "
                "and no post-result alpha or threshold adjustment"
            ),
            selection_context=(
                "Prospective adaptive follow-up after both trained endpoint studies; "
                "fresh confirmation is still unopened"
            ),
        ),
    )


def prepare(root, plan_path):
    p = read(plan_path)
    assert p["alpha_map"] == ALPHAS and p["step"] == 2000
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(root)
    root.mkdir(parents=True)
    (root / "configs").mkdir()
    parents = {tag: Path(e["root"]) for tag, e in p["endpoints"].items()}
    control = Path(p["control"])
    for directory in [*parents.values(), control]:
        original_source(directory)
    a, b = parents["strokes"], parents["mixed"]
    paired_config(
        settings(b / "configs/mixed_source_ink_guard.json", b.parent.parent),
        settings(a / "configs/source_ink_guard.json", a.parent.parent),
        {"sketch": {"mode": "mixed"}},
    )
    configs = {tag: read(parent / "protocol.json") for tag, parent in parents.items()}
    original = configs["strokes"]
    for protocol in [*configs.values(), read(control / "protocol.json")]:
        for key in (
            "initial",
            "initial_files",
            "selected_panels",
            "total_schedule_steps",
            "patch_size",
        ):
            assert protocol[key] == original[key], key
    checks = [read(x / "checkpoint.json") for x in [*parents.values(), control]]
    assert all(x["step"] == 2000 for x in checks)
    assert len({x["clean_sha256"] for x in checks}) == 1
    for name in (
        "dataset.json",
        "evaluation.json",
        "acceptance.json",
        "supplemental-panels.json",
        "confirmation-plan.json",
    ):
        assert read(a / name) == read(b / name) == read(control / name), name
        shutil.copyfile(a / name, root / name)
    assert sha(root / "acceptance.json") == p["acceptance_sha256"]
    assert sha(root / "confirmation-plan.json") == p["confirmation_plan_sha256"]
    shutil.copyfile(plan_path, root / "plan.json")
    config = load_config(a / "configs/source_ink_guard.json")
    save_config(config, root / "config.json")
    endpoints = {}
    for tag, specification in p["endpoints"].items():
        parent, method = parents[tag], specification["method"]
        endpoints[tag] = dict(
            root=str(parent), method=method, source_sha256=sha(parent / "source.json"), modes={}
        )
        for mode in ("ema", "raw"):
            folder = parent / f"snapshots/step-2000/{method}/{mode}"
            files = {n: sha(folder / n) for n in ("weights.npz", "config.json", "metadata.json")}
            assert files == specification["modes"][mode]
            endpoints[tag]["modes"][mode] = dict(path=str(folder), files=files)
    write(
        root / "endpoint-seal.json",
        dict(
            step=2000,
            endpoints=endpoints,
            plan_sha256=sha(plan_path),
            same_training_final_batch_sha256=checks[0]["clean_sha256"],
        ),
    )
    records = []
    for mode in ("ema", "raw"):
        first, c1 = load_inference(endpoints["strokes"]["modes"][mode]["path"])
        second, c2 = load_inference(endpoints["mixed"]["modes"][mode]["path"])
        compatible_config(c1.model, c2.model)
        assert c1.model.source_conditioning and c2.model.source_conditioning
        for name, alpha in ALPHAS.items():
            model = blend_parameters(first, second, alpha)
            destination = root / f"snapshots/step-2000/{name}/{mode}"
            save_inference(
                model,
                config,
                destination,
                provenance=(
                    f"2k guided corruption parameter blend: "
                    f"{1 - alpha} strokes + {alpha} mixture, {mode}. "
                    "No new training or optimizer history."
                ),
            )
            loaded, _ = load_inference(destination)
            jax.tree.map(np.testing.assert_array_equal, nnx.state(model), nnx.state(loaded))
            assert read(destination / "metadata.json")["parameters"] == 126828
            files = {
                n: sha(destination / n) for n in ("weights.npz", "config.json", "metadata.json")
            }
            records.append(
                dict(
                    method=name,
                    alpha=alpha,
                    mode=mode,
                    parameters=126828,
                    roundtrip_exact=True,
                    weights_sha256=files["weights.npz"],
                    files=files,
                )
            )
            save_config(config, root / f"configs/{name}.json")
    baselines = [dict(root=str(control), methods=["control", "ink_guard"])] + [
        dict(root=str(parents[tag]), methods=[p["endpoints"][tag]["method"]])
        for tag in ("strokes", "mixed")
    ]
    for item in baselines:
        item["source_sha256"] = sha(Path(item["root"]) / "source.json")
    protocol = {
        k: original[k]
        for k in (
            "initial",
            "initial_files",
            "selected_panels",
            "total_schedule_steps",
            "patch_size",
        )
    }
    protocol.update(
        methods=list(ALPHAS),
        milestones=[2000],
        alpha_map=ALPHAS,
        step=2000,
        plan_sha256=sha(root / "plan.json"),
        endpoint_seal_sha256=sha(root / "endpoint-seal.json"),
        baselines=baselines,
        study_type="Inference-only checkpoint construction; strict final gates at fixed 2k",
    )
    write(root / "protocol.json", protocol)
    write(
        root / "construction.json",
        dict(
            records=records,
            optimizer_states_created=False,
            script_sha256=sha(Path(__file__)),
            blend_source_sha256=sha(Path("src/lineartgen/models/blending.py")),
        ),
    )
    shutil.copytree(control / "evaluation/step-0-ema", root / "evaluation/step-0-ema")
    freeze(root)
    print(dict(root=str(root), methods=list(ALPHAS), eligibility="strict final"), flush=True)


def validate(root):
    _, endpoints = verify_endpoints(root)
    assert jax.default_backend() != "cpu"
    path = root / "endpoint-validation.json"
    if path.exists():
        assert read(path)["all_forward_values_exact"] and read(path)["source_sha256"] == sha(
            root / "source.json"
        )
        return
    reference = read(root / "evaluation.json")
    cases = Path(reference["root"])
    manifest = read(cases / "cases.json")
    infer = nnx.jit(
        lambda m, x, t, s, n: inference_with_source(m, x, t, s, steps=n), static_argnums=4
    )
    count = 0
    for mode in ("ema", "raw"):
        models = [
            load_inference(endpoints[tag]["modes"][mode]["path"])[0] for tag in ("strokes", "mixed")
        ]
        clones = [blend_parameters(*models, alpha) for alpha in (0, 1)]
        for model, clone in zip(models, clones, strict=True):
            jax.tree.map(np.testing.assert_array_equal, nnx.state(model), nnx.state(clone))
        for index in (0, 480, 1295):
            case = manifest["cases"][index]
            file = cases / case["file"]
            assert sha(file) == reference["files"][case["file"]]
            with np.load(file, allow_pickle=False) as arrays:
                source = arrays["source"] if "source" in arrays else arrays["image"]
                strength = arrays["strength"]
            prepared = jax.device_put(
                runtime_inputs(source[None], strength[None], case["seed"], levels=3)
            )
            source = jax.device_put(source[None])
            for steps in (1, 2, 5):
                for model, clone in zip(models, clones, strict=True):
                    np.testing.assert_array_equal(
                        np.asarray(infer(model, *prepared, source, steps)),
                        np.asarray(infer(clone, *prepared, source, steps)),
                    )
                    count += 1
    write(
        path,
        dict(
            all_forward_values_exact=True,
            comparisons=count,
            indices=[0, 480, 1295],
            devices=[str(d) for d in jax.devices()],
            source_sha256=sha(root / "source.json"),
        ),
    )
    print("Guided GPU endpoint checks exact", count, flush=True)


def compare(root):
    protocol, _ = verify_endpoints(root)
    reference = read(root / "evaluation.json")
    manifest = read(Path(reference["root"]) / "cases.json")
    assert sha(Path(reference["root"]) / "cases.json") == reference["cases_sha256"]
    initial, artifacts = audit(root, 0, "ema", reference, len(manifest["cases"]))
    parents = [(Path(d["root"]), d["methods"]) for d in protocol["baselines"]]
    for parent, _ in parents:
        declaration = next(d for d in protocol["baselines"] if Path(d["root"]) == parent)
        assert sha(parent / "source.json") == declaration["source_sha256"]
        original_source(parent)
        previous, files = audit(parent, 0, "ema", reference, len(manifest["cases"]))
        assert previous["results"] == initial["results"]
        artifacts.update(files)
    methods, milestones, gates = protocol["methods"], {}, {}
    basenames = [n for _, names in parents for n in names]
    for mode in ("ema", "raw"):
        own, files = audit(root, 2000, mode, reference, len(manifest["cases"]))
        artifacts.update(files)
        rows = own["results"].copy()
        for parent, names in parents:
            before, files = audit(parent, 2000, mode, reference, len(manifest["cases"]))
            artifacts.update(files)
            for name in names:
                baseline = [r for r in before["results"] if r["method"] == name]
                align_rows(own["results"], baseline, methods)
                rows.extend(baseline)
        summary = summarize(rows)
        milestones[f"2000-{mode}"] = dict(
            summary=summary,
            paired=[r for name in basenames for r in differences(rows, rows, manifest, name)]
            + differences(rows, initial["results"], manifest, "initial"),
        )
        gates[mode] = {
            phase: [
                assess(summary, read(root / "acceptance.json"), n, screening=flag) for n in methods
            ]
            for phase, flag in [("screen", True), ("final", False)]
        }
    analysis = root / "analysis-with-baselines-2000.json"
    write(
        analysis,
        dict(
            through=2000,
            milestones=milestones,
            artifacts=artifacts,
            limitations=(
                "Adaptive inference-only development study; "
                "no independent seed or fresh confirmation yet"
            ),
            analysis_source_sha256=sha(Path(__file__)),
        ),
    )
    # A 2k checkpoint is a fixed fitted model here, not eligibility for more training.
    # All unchanged FINAL tolerances apply before allowing independent confirmation.
    results = gates["ema"]["final"]
    ranked = sorted(
        [r for r in results if r["numerical_gate_passed"]],
        key=lambda r: -sum(r["primary_halo_gains"].values()),
    )
    write(
        root / "quality-gate-with-baselines-2000.json",
        dict(
            step=2000,
            results=results,
            ranked_provisional_candidates=[r["method"] for r in ranked],
            diagnostics=gates,
            eligibility="strict final despite 2k endpoint age",
            analysis_sha256=sha(analysis),
            rules_sha256=sha(root / "acceptance.json"),
        ),
    )
    panels, selected = {}, [*basenames, *methods]
    for suite in ("evaluation", "supplemental-evaluation"):
        for mode in ("ema", "raw"):
            for suffix in ("", "-diagnostic") if suite == "evaluation" else ("",):
                relative = Path(suite) / f"step-2000-{mode}{suffix}"
                for path in sorted((root / relative).glob("*-krita-white.png")):
                    sources = [(path, methods)] + [
                        (p / relative / path.name, read(p / "protocol.json")["methods"])
                        for p, _ in parents
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
    print(
        [(r["method"], r["numerical_gate_passed"], len(r["failures"])) for r in results], flush=True
    )


def run(root):
    verify_endpoints(root)
    validate(root)
    for mode in ("ema", "raw"):
        evaluate(root, 2000, mode)
    evaluate(root, 2000, "ema", diagnostic=True)
    for mode in ("ema", "raw"):
        for script in ("evaluate_lineart_blank_study", "render_lineart_loss_supplement"):
            subprocess.run(
                [
                    sys.executable,
                    f"scripts/{script}.py",
                    "--root",
                    str(root),
                    "--step",
                    "2000",
                    "--mode",
                    mode,
                ],
                check=True,
            )
    compare(root)
    write(root / "workflow-status.json", dict(stage="numerical_complete_visual_review_pending"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "prepare", "run"])
    for name in ("root", "plan", "strokes", "mixed", "control"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if args.action == "plan":
        plan(
            args.plan.resolve(),
            args.strokes.resolve(),
            args.mixed.resolve(),
            args.control.resolve(),
        )
    elif args.action == "prepare":
        prepare(args.root.resolve(), args.plan.resolve())
    else:
        run(args.root.resolve())
