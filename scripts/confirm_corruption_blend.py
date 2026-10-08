"""Seal passing fixed-alpha EMA models, then evaluate prospective synthetic draws.

This script does not generate the confirmation inputs. Use the already frozen
prepare_ink_structure_confirmation.py only after the selection seal exists.
"""

import argparse
import subprocess
from pathlib import Path

import jax
import numpy as np
from assess_ink_structure_study import assess
from campaign_protocol import read, sha, write
from evaluate_lineart_loss_study import render
from flax import nnx
from prepare_ink_structure_confirmation import check_selection
from summarize_lineart_loss_study import differences, summarize

from lineartgen.evaluation import edit_boundary_metrics, probe_metrics, protected_metrics
from lineartgen.evaluation_sampling import background_metrics, inference_with_source, runtime_inputs
from lineartgen.lineart_metrics import halo_metrics
from lineartgen.models.serialization import load_inference


def seal(root, primary, secondary):
    """Require strict development gates and an actual recorded visual review for both seeds."""
    destination = root / "confirmation-selection.json"
    if destination.exists():
        raise FileExistsError(destination)
    evidence = {}
    for parent, name in ((primary, "corruption_blend0625"), (secondary, "candidate")):
        gate_path = parent / "quality-gate-with-baselines-2000.json"
        gate, review = read(gate_path), read(parent / "review.json")
        result = next(r for r in gate["results"] if r["method"] == name)
        assert result["numerical_gate_passed"] and not result["screening"]
        assert review["approved_methods"] == [name] and review["weights"] == "ema"
        assert review["selected_alpha"] == 1 / 16
        assert review["quality_gate_sha256"] == sha(gate_path)
        assert review["inspection_notes"] and review["inspected_panels"]
        for path, digest in review["inspected_panels"].items():
            assert sha(path) == digest
        for filename in ("quality-gate-with-baselines-2000.json", "review.json", "source.json"):
            path = parent / filename
            evidence[str(path)] = sha(path)
        for filename in ("confirmation-plan.json", "acceptance.json", "dataset.json"):
            assert read(parent / filename) == read(root / filename)
    control = Path(read(primary / "plan.json")["control"])
    paths = {
        "candidate_07": primary / "snapshots/step-2000/corruption_blend0625/ema",
        "control_07": control / "snapshots/step-2000/control/ema",
        "candidate_06": secondary / "snapshots/step-2000/candidate/ema",
        "control_06": secondary / "snapshots/step-2000/control/ema",
    }
    tracked = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"], text=True
    ).splitlines()
    source = {
        str(Path(p).resolve()): sha(p)
        for p in tracked
        if p.endswith((".py", ".toml")) or p == "uv.lock"
    }
    assert str(Path(__file__).resolve()) in source
    write(
        destination,
        dict(
            confirmation_plan_sha256=sha(root / "confirmation-plan.json"),
            acceptance_sha256=sha(root / "acceptance.json"),
            config_sha256=sha(root / "config.json"),
            dataset_sha256=sha(root / "dataset.json"),
            source_files=source,
            evidence=evidence,
            alpha=1 / 16,
            step=2000,
            weights="ema",
            primary="candidate_07",
            pairs={"candidate_07": "control_07", "candidate_06": "control_06"},
            models={
                name: dict(
                    path=str(path),
                    files={
                        n: sha(path / n) for n in ("weights.npz", "metadata.json", "config.json")
                    },
                )
                for name, path in paths.items()
            },
            selection_notes=(
                "Fixed alpha=1/16, EMA, exact 2k endpoints on the original 10k LR clock. "
                "Both independent fine-tuning seeds passed strict development gates and visual "
                "review. Seed20261007 remains primary; no choice between seeds or retuning on "
                "these prospective draws. The 64 source images have been seen in earlier "
                "evaluation; only crops and synthetic draws are fresh."
            ),
            evaluation_policy=dict(
                primary_denoise_steps=1,
                diagnostic_denoise_steps=[2, 5],
                zero_strength="Exact source identity only; excluded from active quality gates",
                decision="Both seed pairs must pass unchanged strict final criteria at one step",
                panel_rule=(
                    "First four real sources and all eight geometry kinds at variant40; "
                    "whole strong tone/blur/strokes at strengths .1/.5, plus clean .1 partial "
                    "and every selected target's zero-strength control; all 1/2/5 steps"
                ),
            ),
        ),
    )
    verify_seal(root)
    print(dict(sealed=str(destination), fresh_inputs_generated=False), flush=True)


def verify_seal(root):
    path = root / "confirmation-selection.json"
    selection = check_selection(path, root / "confirmation-plan.json")
    for name in ("acceptance", "config", "dataset"):
        assert sha(root / f"{name}.json") == selection[f"{name}_sha256"]
    for group in ("source_files", "evidence"):
        for name, digest in selection[group].items():
            assert sha(name) == digest, name
    return selection


def active_rows(rows, steps):
    """Never mix refinement counts, nor include protected-only identity controls."""
    return [r for r in rows if r["strength"] > 0 and r["denoise_steps"] == steps]


def aggregate(rows, rules, pairs, steps):
    summary = summarize(active_rows(rows, steps))
    gates = {
        candidate: assess(summary, dict(rules, primary_baseline=baseline), candidate)
        for candidate, baseline in pairs.items()
    }
    return dict(summary=summary, gates=gates)


def panel_indices(manifest):
    real = list(dict.fromkeys(c["source"] for c in manifest["cases"] if c["kind"] == "real"))[:4]
    return [
        i
        for i, c in enumerate(manifest["cases"])
        if (c.get("source") in real or c.get("variant") == 40)
        and (
            c["strength"] == 0
            or (not c["partial"] and c["severity"] == "strong")
            or (c["family"] == "clean" and c["partial"] and c["strength"] == 0.1)
        )
    ]


def validate_rows(rows, case, index, methods, counts):
    expected = {(m, s) for m in methods for s in counts}
    actual = [(r["method"], r["denoise_steps"]) for r in rows]
    assert len(actual) == len(expected) and set(actual) == expected
    for row in rows:
        assert row["case"] == index and row["strength"] == case["strength"]
        assert row["protected_max_error"] in (None, 0.0)
        assert all(np.isfinite(v) for v in row.values() if isinstance(v, (int, float)))
        if case["strength"] == 0:
            assert row["protected_max_error"] == 0.0


def evaluate(root, cases, destination):
    selection = verify_seal(root)
    manifest, inputs = read(cases / "cases.json"), read(cases / "input-sha256.json")
    assert manifest["selection_seal_sha256"] == sha(root / "confirmation-selection.json")
    assert manifest["confirmation_plan_sha256"] == selection["confirmation_plan_sha256"]
    assert inputs["cases.json"] == sha(cases / "cases.json")
    plan = read(root / "confirmation-plan.json")
    counts = plan["denoise_steps"]
    assert counts == [1, 2, 5]
    assert len(manifest["cases"]) == 2784
    binding = dict(
        selection_sha256=sha(root / "confirmation-selection.json"),
        input_index_sha256=sha(cases / "input-sha256.json"),
        evaluator_sha256=sha(Path(__file__)),
        selected_panels=panel_indices(manifest),
    )
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "binding.json").exists():
        assert read(destination / "binding.json") == binding
    else:
        write(destination / "binding.json", binding)
    models = {name: load_inference(m["path"])[0] for name, m in selection["models"].items()}
    infer = nnx.jit(
        lambda m, x, t, source, s: inference_with_source(m, x, t, source, steps=s), static_argnums=4
    )
    rows, shards = [], {}
    for index, case in enumerate(manifest["cases"]):
        path = cases / case["file"]
        assert sha(path) == inputs[case["file"]]
        shard_path = destination / f"cases/case-{index:04d}.json"
        if shard_path.exists():
            shard = read(shard_path)
            assert shard["binding_sha256"] == sha(destination / "binding.json")
            assert shard["input_sha256"] == inputs[case["file"]]
            for panel, digest in shard["panels"].items():
                assert sha(destination / panel) == digest
            records = shard["results"]
        else:
            with np.load(path, allow_pickle=False) as arrays:
                clean, source, strength = arrays["image"], arrays["source"], arrays["strength"]
                target = np.where(strength > 0, clean, source)
                probes = arrays["probes"] if "probes" in arrays else None
                connections = arrays["connections"] if "connections" in arrays else None
            prepared = runtime_inputs(source[None], strength[None], case["seed"], levels=3)
            noisy = prepared[0][0][0]
            prepared = jax.device_put(prepared)
            guide = jax.device_put(source[None])
            records, panels = [], {}
            for iterations in counts:
                predictions = {}
                for name, model in models.items():
                    prediction = np.asarray(infer(model, *prepared, guide, iterations))[0]
                    assert np.isfinite(prediction).all(), (index, name, iterations)
                    extra = (
                        probe_metrics(prediction, target, probes, connections)
                        if probes is not None and not case["partial"] and case["strength"] > 0
                        else {}
                    )
                    measured = (
                        protected_metrics(prediction, target, strength)
                        | background_metrics(prediction, target, strength)
                        | edit_boundary_metrics(prediction, target, strength)
                        | halo_metrics(prediction, target, strength)
                        | extra
                    )
                    records.append(
                        dict(
                            case=index,
                            method=name,
                            domain="real" if case["kind"] == "real" else "geometry",
                            kind=case["kind"],
                            source=case.get("source"),
                            suite=case["suite"],
                            family=case["family"],
                            severity=case["severity"],
                            strength=case["strength"],
                            partial=case["partial"],
                            denoise_steps=iterations,
                            **measured,
                        )
                    )
                    predictions[name] = prediction
                if index in binding["selected_panels"]:
                    panel_folder = destination / "panels"
                    panel_folder.mkdir(exist_ok=True)
                    render(panel_folder, index, iterations, target, source, noisy, predictions)
                    for display in ("continuous", "krita-white"):
                        p = panel_folder / f"case-{index:04d}-steps-{iterations}-{display}.png"
                        panels[str(p.relative_to(destination))] = sha(p)
            write(
                shard_path,
                dict(
                    binding_sha256=sha(destination / "binding.json"),
                    input_sha256=inputs[case["file"]],
                    results=records,
                    panels=panels,
                ),
            )
        validate_rows(records, case, index, models, counts)
        rows.extend(records)
        shards[str(shard_path.relative_to(destination))] = sha(shard_path)
        if (index + 1) % 100 == 0 or index + 1 == len(manifest["cases"]):
            print(f"Confirmation {index + 1}/{len(manifest['cases'])}", flush=True)
    verify_seal(root)
    for name, digest in inputs.items():
        assert sha(cases / name) == digest
    results = {
        str(s): aggregate(rows, read(root / "acceptance.json"), selection["pairs"], s)
        for s in counts
    }
    # Paired CIs are conditional on these fitted models; do not pool the two seeds.
    for s in counts:
        current = active_rows(rows, s)
        paired = []
        for candidate, baseline in selection["pairs"].items():
            pair = [r for r in current if r["method"] in (candidate, baseline)]
            paired.extend(differences(pair, pair, manifest, baseline))
        results[str(s)]["paired"] = paired
    zero = [r for r in rows if r["strength"] == 0]
    assert len(zero) == 96 * len(models) * len(counts)
    write(
        destination / "analysis.json",
        dict(
            binding=binding,
            case_shards=shards,
            prediction_count=len(rows),
            zero_strength_count=len(zero),
            all_protected_pixels_exact=True,
            iterations=results,
            primary_denoise_steps=1,
            both_seeds_pass=all(g["numerical_gate_passed"] for g in results["1"]["gates"].values()),
            limitations=manifest["limitations"],
            status="Numerical analysis only; native-pixel visual review remains required",
        ),
    )
    print(
        {
            k: (v["numerical_gate_passed"], len(v["failures"]))
            for k, v in results["1"]["gates"].items()
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["seal", "evaluate"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--primary", type=Path)
    parser.add_argument("--secondary", type=Path)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    if args.action == "seal":
        seal(args.root.resolve(), args.primary.resolve(), args.secondary.resolve())
    else:
        evaluate(args.root.resolve(), args.cases.resolve(), args.destination.resolve())
