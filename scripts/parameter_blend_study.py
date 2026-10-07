"""Fixed checkpoint interpolation study; no optimizer or new training trajectory."""

import argparse
import importlib.metadata
import shutil
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
from assess_ink_structure_study import assess
from campaign_protocol import read, sha, write
from compare_output_gradient_study import align_rows
from compare_paper_guard_study import combine_panels, paired_config, settings
from evaluate_lineart_loss_study import evaluate
from flax import nnx
from summarize_lineart_loss_study import audit, differences, summarize
from train_lineart_loss_study import verify

from lineartgen.config import load_config, save_config
from lineartgen.diffusion import sample_prepared
from lineartgen.evaluation_sampling import runtime_inputs
from lineartgen.models.blending import blend_parameters, compatible_config
from lineartgen.models.serialization import load_inference, save_inference


def original_source(root):
    source = read(root / "source.json")
    for name, digest in source["files"].items():
        assert sha(root.parent.parent / name) == digest, name
    for name, digest in source["settings"].items():
        assert sha(root / name) == digest, name


def freeze(root):
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    tracked = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    paths = [Path(p) for p in tracked if p.endswith((".py", ".toml")) or p == "uv.lock"]
    subprocess.run(
        ["git", "archive", "--format=tar.gz", "-o", str(root / "source.tar.gz"), commit], check=True
    )
    settings_files = sorted(root.glob("*.json")) + sorted((root / "configs").glob("*.json"))
    write(
        root / "source.json",
        dict(
            commit=commit,
            files={str(p): sha(p) for p in paths},
            settings={str(p.relative_to(root)): sha(p) for p in settings_files},
            archive_sha256=sha(root / "source.tar.gz"),
            versions={n: importlib.metadata.version(n) for n in ["jax", "jaxlib", "flax", "numpy"]},
            type="Inference checkpoint construction only; no generator updates or optimizer states",
            construction_sha256=sha(root / "construction.json"),
        ),
    )


def prepare(root, plan_root, step, review):
    plan_path = plan_root / "planned-parameter-blends.json"
    plan = read(plan_path)
    gate = read(plan_root / "quality-gate-with-baselines-10000.json")
    assert not gate["ranked_provisional_candidates"], "Confirm a passing trained model first"
    assert plan["alphas"] == [0.25, 0.5, 0.75] and step in (2000, 10000)
    alphas = {f"blend{int(a * 100)}": a for a in plan["alphas"]}
    if step == 10000:
        decision = read(review)
        screen = Path(decision["screen_root"])
        assert decision["quality_gate_sha256"] == sha(
            screen / "quality-gate-with-baselines-2000.json"
        )
        assert decision["plan_sha256"] == sha(plan_path)
        assert decision["inspection_notes"] and decision["inspected_panels"]
        for p, digest in decision["inspected_panels"].items():
            assert sha(p) == digest
        eligible = read(screen / "quality-gate-with-baselines-2000.json")[
            "ranked_provisional_candidates"
        ]
        names = decision["approved_methods"]
        assert names and len(set(names)) == len(names) and set(names) <= set(eligible)
        alphas = {n: alphas[n] for n in names}
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(root)
    root.mkdir(parents=True)
    (root / "configs").mkdir()
    hard = Path(
        plan["endpoints_2000"]["hard"]["root"]
        if step == 2000
        else plan["endpoints_10000"]["hard_root"]
    )
    ste = Path(
        plan["endpoints_2000"]["ste"]["root"]
        if step == 2000
        else plan["endpoints_10000"]["ste_root"]
    )
    for parent in (hard, ste):
        original_source(parent)
    protocols = [read(p / "protocol.json") for p in (hard, ste)]
    for key in (
        "initial",
        "initial_files",
        "selected_panels",
        "total_schedule_steps",
        "patch_size",
    ):
        assert protocols[0][key] == protocols[1][key], key
    checkpoints = [read(p / "checkpoint.json") for p in (hard, ste)]
    assert all(c["step"] == step for c in checkpoints)
    assert checkpoints[0]["clean_sha256"] == checkpoints[1]["clean_sha256"]
    paired_config(
        settings(ste / "configs/ste_neighbor_guard.json", ste.parent.parent),
        settings(hard / "configs/neighbor_guard.json", hard.parent.parent),
        {"model": {"output_clip_gradient": "straight_through"}},
    )
    for name in (
        "dataset.json",
        "evaluation.json",
        "acceptance.json",
        "supplemental-panels.json",
        "confirmation-plan.json",
    ):
        assert read(hard / name) == read(ste / name) == read(plan_root / name)
        shutil.copyfile(plan_root / name, root / name)
    assert plan["acceptance_sha256"] == sha(root / "acceptance.json")
    assert plan["confirmation_plan_sha256"] == sha(root / "confirmation-plan.json")
    shutil.copyfile(plan_path, root / "plan.json")
    if review:
        shutil.copyfile(review, root / "selection-review.json")
    config = load_config(hard / "configs/neighbor_guard.json")
    config = config.model_copy(
        update={
            "data": config.data.model_copy(
                update={
                    k: (hard.parent.parent / getattr(config.data, k)).resolve()
                    for k in ("root", "prepared_root")
                    if getattr(config.data, k) is not None
                }
            )
        }
    )
    save_config(config, root / "config.json")
    endpoints = {}
    for tag, parent, method in [
        ("hard", hard, "neighbor_guard"),
        ("ste", ste, "ste_neighbor_guard"),
    ]:
        endpoints[tag] = dict(
            root=str(parent), method=method, source_sha256=sha(parent / "source.json"), modes={}
        )
        for mode in ("ema", "raw"):
            path = parent / f"snapshots/step-{step}/{method}/{mode}"
            files = {n: sha(path / n) for n in ("weights.npz", "config.json", "metadata.json")}
            if step == 2000:
                assert files["weights.npz"] == plan["endpoints_2000"][tag]["files"][mode]
            endpoints[tag]["modes"][mode] = dict(path=str(path), files=files)
    write(
        root / "endpoint-seal.json",
        dict(
            step=step,
            endpoints=endpoints,
            plan_sha256=sha(plan_path),
            condition_gate_sha256=sha(plan_root / "quality-gate-with-baselines-10000.json"),
            same_training_final_batch_sha256=checkpoints[0]["clean_sha256"],
        ),
    )
    records = []
    for mode in ("ema", "raw"):
        first, a = load_inference(endpoints["hard"]["modes"][mode]["path"])
        second, b = load_inference(endpoints["ste"]["modes"][mode]["path"])
        compatible_config(a.model, b.model)
        for name, alpha in alphas.items():
            model = blend_parameters(first, second, alpha)
            destination = root / f"snapshots/step-{step}/{name}/{mode}"
            save_inference(
                model,
                config,
                destination,
                provenance=f"Post-training parameter blend: "
                f"{alpha} STE neighbor + {1 - alpha} hard neighbor at step {step}; {mode}. "
                "No optimizer history and no new training updates.",
            )
            loaded, _ = load_inference(destination)
            jax.tree.map(np.testing.assert_array_equal, nnx.state(model), nnx.state(loaded))
            assert read(destination / "metadata.json")["parameters"] == 126765
            records.append(
                dict(
                    method=name,
                    alpha=alpha,
                    mode=mode,
                    parameters=126765,
                    roundtrip_exact=True,
                    weights_sha256=sha(destination / "weights.npz"),
                    files={
                        n: sha(destination / n)
                        for n in ("weights.npz", "config.json", "metadata.json")
                    },
                )
            )
            save_config(config, root / f"configs/{name}.json")
    protocol = {
        k: protocols[0][k]
        for k in (
            "initial",
            "initial_files",
            "selected_panels",
            "total_schedule_steps",
            "patch_size",
        )
    }
    protocol.update(
        methods=list(alphas),
        milestones=[step],
        alpha_map=alphas,
        step=step,
        plan_sha256=sha(root / "plan.json"),
        endpoint_seal_sha256=sha(root / "endpoint-seal.json"),
        study_type="Parameter interpolation, not training or output averaging",
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
    shutil.copytree(hard / "evaluation/step-0-ema", root / "evaluation/step-0-ema")
    freeze(root)
    print(dict(root=str(root), step=step, methods=list(alphas)), flush=True)


def verify_endpoints(root):
    protocol = verify(root)
    assert sha(root / "plan.json") == protocol["plan_sha256"]
    assert sha(root / "endpoint-seal.json") == protocol["endpoint_seal_sha256"]
    endpoints = read(root / "endpoint-seal.json")["endpoints"]
    for endpoint in endpoints.values():
        parent = Path(endpoint["root"])
        assert sha(parent / "source.json") == endpoint["source_sha256"]
        original_source(parent)
        for value in endpoint["modes"].values():
            for name, digest in value["files"].items():
                assert sha(Path(value["path"]) / name) == digest
    for record in read(root / "construction.json")["records"]:
        folder = root / f"snapshots/step-{protocol['step']}" / record["method"] / record["mode"]
        for name, digest in record["files"].items():
            assert sha(folder / name) == digest
    return protocol, endpoints


def validate(root):
    protocol, endpoints = verify_endpoints(root)
    assert jax.default_backend() != "cpu", "Use the same GPU backend as the recorded baselines"
    path = root / "endpoint-validation.json"
    if path.exists():
        prior = read(path)
        assert prior["source_sha256"] == sha(root / "source.json")
        assert prior["all_forward_values_exact"]
        return
    reference = read(root / "evaluation.json")
    cases = Path(reference["root"])
    manifest = read(cases / "cases.json")
    infer = nnx.jit(lambda m, x, t, n: sample_prepared(m, x, t, steps=n), static_argnums=3)
    count = 0
    for mode in ("ema", "raw"):
        models = [
            load_inference(endpoints[tag]["modes"][mode]["path"])[0] for tag in ("hard", "ste")
        ]
        clones = [blend_parameters(*models, alpha) for alpha in (0, 1)]
        for m, c in zip(models, clones, strict=True):
            jax.tree.map(np.testing.assert_array_equal, nnx.state(m), nnx.state(c))
        for index in (0, 480, 1295):
            case = manifest["cases"][index]
            p = cases / case["file"]
            assert sha(p) == reference["files"][case["file"]]
            with np.load(p, allow_pickle=False) as arrays:
                clean, t = arrays["image"], arrays["strength"]
                source = arrays["source"] if "source" in arrays else clean
            prepared = jax.device_put(runtime_inputs(source[None], t[None], case["seed"], levels=3))
            for n in (1, 2, 5):
                for m, c in zip(models, clones, strict=True):
                    np.testing.assert_array_equal(
                        np.asarray(infer(m, *prepared, n)), np.asarray(infer(c, *prepared, n))
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
    print("GPU endpoint validation exact", count, flush=True)


def compare(root):
    protocol, endpoints = verify_endpoints(root)
    step = protocol["step"]
    methods = protocol["methods"]
    reference = read(root / "evaluation.json")
    assert sha(Path(reference["root"]) / "cases.json") == reference["cases_sha256"]
    manifest = read(Path(reference["root"]) / "cases.json")
    initial, artifacts = audit(root, 0, "ema", reference, len(manifest["cases"]))
    parents = [
        (Path(endpoints["hard"]["root"]), ["control", "neighbor_guard"]),
        (Path(endpoints["ste"]["root"]), ["ste_neighbor_guard"]),
    ]
    for parent, _ in parents:
        p, files = audit(parent, 0, "ema", reference, len(manifest["cases"]))
        assert p["results"] == initial["results"]
        artifacts.update(files)
    milestones = {}
    gates = {}
    for mode in ("ema", "raw"):
        own, files = audit(root, step, mode, reference, len(manifest["cases"]))
        artifacts.update(files)
        rows = own["results"].copy()
        for parent, names in parents:
            before, files = audit(parent, step, mode, reference, len(manifest["cases"]))
            artifacts.update(files)
            for name in names:
                baseline = [r for r in before["results"] if r["method"] == name]
                align_rows(own["results"], baseline, methods)
                rows.extend(baseline)
        summary = summarize(rows)
        milestones[f"{step}-{mode}"] = dict(
            summary=summary,
            paired=[
                r
                for name in ["control", "neighbor_guard", "ste_neighbor_guard"]
                for r in differences(rows, rows, manifest, name)
            ],
        )
        gates[mode] = {
            phase: [
                assess(summary, read(root / "acceptance.json"), m, screening=flag) for m in methods
            ]
            for phase, flag in [("screen", True), ("final", False)]
        }
    analysis = root / f"analysis-with-baselines-{step}.json"
    write(
        analysis,
        dict(
            through=step,
            milestones=milestones,
            artifacts=artifacts,
            limitations="Post-training interpolation; no independent seed. "
            "Synthetic development only.",
            analysis_source_sha256=sha(Path(__file__)),
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
        ),
    )
    panels = {}
    selected = ["control", "neighbor_guard", "ste_neighbor_guard", *methods]
    for suite in ("evaluation", "supplemental-evaluation"):
        for mode in ("ema", "raw"):
            for suffix in ("", "-diagnostic") if suite == "evaluation" else ("",):
                relative = Path(suite) / f"step-{step}-{mode}{suffix}"
                for source in sorted((root / relative).glob("*-krita-white.png")):
                    sources = [(source, methods)] + [
                        (p / relative / source.name, read(p / "protocol.json")["methods"])
                        for p, _ in parents
                    ]
                    if all(p.exists() for p, _ in sources):
                        panels.update(
                            combine_panels(
                                sources,
                                root / "comparison-panels" / relative / source.name,
                                selected,
                                initial=suite == "supplemental-evaluation",
                            )
                        )
    write(root / f"comparison-panels-{step}.json", dict(files=panels, labels=selected))
    print(
        [(r["method"], r["numerical_gate_passed"], len(r["failures"])) for r in results], flush=True
    )


def run(root):
    protocol, _ = verify_endpoints(root)
    step = protocol["step"]
    validate(root)
    for mode in ("ema", "raw"):
        evaluate(root, step, mode)
    evaluate(root, step, "ema", diagnostic=True)
    for mode in ("ema", "raw"):
        for script in ("evaluate_lineart_blank_study", "render_lineart_loss_supplement"):
            subprocess.run(
                [
                    sys.executable,
                    f"scripts/{script}.py",
                    "--root",
                    str(root),
                    "--step",
                    str(step),
                    "--mode",
                    mode,
                ],
                check=True,
            )
    compare(root)
    write(root / "workflow-status.json", dict(stage="numerical_complete_visual_review_pending"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-root", type=Path)
    parser.add_argument("--step", type=int, choices=[2000, 10000])
    parser.add_argument("--review", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root.resolve(), args.plan_root.resolve(), args.step, args.review)
    else:
        run(args.root.resolve())
