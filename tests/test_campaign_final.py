import importlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def module(name, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module(name)


def test_paired_intervals_weight_cases_and_resample_fitted_seeds(monkeypatch):
    analysis = module("analyze_completion_campaign", monkeypatch)
    manifest = dict(cases=[dict(kind="real", source=s) for s in ("a", "a", "a", "b")])
    pairs = []
    for offset in (-0.1, 0.0, 0.1):
        left, right = [], []
        for i, delta in enumerate((-0.2, -0.2, -0.2, 0.2)):
            row = dict(case=i, strength=0.5, suite="matched_real", active_target_ink_fraction=0.1)
            left.append(dict(row, active_ink_f1=0.5 + delta + offset))
            right.append(dict(row, active_ink_f1=0.5))
        pairs.append((left, right))
    result = analysis.interval(pairs, manifest, "clean_real", "active_ink_f1")
    assert result["delta"] == pytest.approx(-0.1)
    assert result["per_seed"] == pytest.approx([-0.2, -0.1, 0.0])
    assert (result["seeds"], result["source_groups"], result["paired_cases"]) == (3, 2, 12)
    assert result["hierarchical_95_interval"][0] < -0.2
    assert result["hierarchical_95_interval"][1] > 0.0
    same = analysis.interval([(r, r) for _, r in pairs], manifest, "clean_real", "active_ink_f1")
    assert same["hierarchical_95_interval"] == [0.0, 0.0]
    pairs[0][0][0]["active_ink_f1"] = None
    with pytest.raises(AssertionError):
        analysis.interval(pairs, manifest, "clean_real", "active_ink_f1")


def test_random_blank_crops_cannot_hide_line_scores_in_diagnostics(monkeypatch):
    analysis = module("analyze_completion_campaign", monkeypatch)
    shared = dict(suite="sketch_real", family="strokes", severity="strong", strength=0.5)
    result = analysis.real_ink_summaries(
        [
            dict(shared, active_target_ink_fraction=0.0, active_ink_f1=1.0),
            dict(shared, active_target_ink_fraction=0.1, active_ink_f1=0.2),
            dict(shared, active_target_ink_fraction=None, active_ink_f1=None, strength=0.0),
        ]
    )
    indexed = {r["population"]: r for r in result}
    assert indexed["all"]["cases"] == 2
    assert indexed["all"]["active_ink_f1"] == pytest.approx(0.6)
    assert indexed["with_ink"]["active_ink_f1"] == 0.2
    assert indexed["low_ink"]["active_ink_f1"] == 1.0


def test_final_data_cannot_be_opened_before_seal_and_changed_artifacts_reject(
    tmp_path, monkeypatch
):
    final = module("prepare_campaign_final", monkeypatch)
    analysis = module("analyze_completion_campaign", monkeypatch)
    opened = []
    monkeypatch.setattr(final, "discover", lambda *a: opened.append(True))
    with pytest.raises(FileNotFoundError):
        final.prepare(tmp_path)
    assert not opened
    artifact = tmp_path / "weights"
    artifact.write_bytes(b"original")
    final.write(
        tmp_path / "selection-seal.json", dict(artifacts={str(artifact): final.sha(artifact)})
    )
    assert analysis.check_seal(tmp_path)["artifacts"]
    artifact.write_bytes(b"changed")
    with pytest.raises(AssertionError, match="Sealed selection changed"):
        final.prepare(tmp_path)
    assert not opened


def test_final_manifest_keeps_all_sources_blanks_and_disjoint_split(tmp_path, monkeypatch):
    final = module("prepare_campaign_final", monkeypatch)
    from lineartgen.config import ExperimentConfig, save_config

    final.write(tmp_path / "selection-seal.json", dict(artifacts={}, selected_model=None))
    protocol = final.read(Path(__file__).parents[1] / "configs/completion-protocol.json")
    final.write(tmp_path / "protocol.json", protocol)
    save_config(ExperimentConfig(), tmp_path / "config.json")
    sources = [
        SimpleNamespace(name=f"held-{i}", variants=[(tmp_path / str(i), 512, 512)] * 6)
        for i in range(64)
    ]
    train, dev = [SimpleNamespace(name="train")], [SimpleNamespace(name="dev")]
    monkeypatch.setattr(final, "discover", lambda *a: sources)
    monkeypatch.setattr(final, "split_experiment", lambda *a: (train, dev, sources))
    monkeypatch.setattr(final, "dataset_description", lambda *a: {"fixture": True})
    decoded = []

    def read_crop(path, x, y, size, **kwargs):
        decoded.append(path.name)
        return np.ones((512, 512, 1), np.float32)

    monkeypatch.setattr(final, "read_crop", read_crop)
    monkeypatch.setattr(final.jax, "jit", lambda f: f)
    monkeypatch.setattr(final, "augment_sketch", lambda clean, key, **kw: clean * 0.5)
    monkeypatch.setattr(final, "line_case", lambda *a: (np.ones((512, 512, 1), np.float32), [], []))

    def store(path, **arrays):
        assert arrays["image"].shape == (512, 512, 1)
        path.write_bytes(b"fixture; actual array storage is covered by evaluation tests")

    monkeypatch.setattr(final.np, "savez_compressed", store)
    folder = final.prepare(tmp_path)
    manifest = final.read(folder / "cases.json")
    assert len(decoded) == 64 and set(decoded) == {str(i) for i in range(64)}
    assert len(manifest["cases"]) == 1632 and len(manifest["targets"]) == 96
    assert all(t["ink_fraction"] == 0 for t in manifest["targets"])
    assert len([c for c in manifest["cases"] if c["strength"] == 0]) == 96
    assert len({c["seed"] for c in manifest["cases"]}) == 1632
    assert final.prepare(tmp_path) == folder and len(decoded) == 64
    (folder / "case-000.npz").write_bytes(b"tamper")
    with pytest.raises(AssertionError):
        final.prepare(tmp_path)


def test_blind_plan_balances_sources_modes_and_seeds_before_results(monkeypatch):
    renderer = module("render_campaign_blind", monkeypatch)
    protocol = renderer.read(Path(__file__).parents[1] / "configs/completion-protocol.json")
    plan = renderer.panel_plan(protocol)
    assert plan == renderer.panel_plan(protocol)
    assert len(plan) == len({p["case"] for p in plan}) == 32
    assert sum(p["case"] < 64 * 17 for p in plan) == 16
    assert sum(p["mode"] == "ema" for p in plan) == 24
    assert {p["model_seed"] for p in plan} == {"42", "43", "44"}
    assert all(set(p["order"]) == set(renderer.MODELS) for p in plan)
    assert len({tuple(p["order"]) for p in plan}) > 20


def test_saved_parameter_counts_agree_with_extracted_models(tmp_path, monkeypatch):
    analysis = module("analyze_completion_campaign", monkeypatch)
    import jax
    from flax import nnx

    from lineartgen.config import ExperimentConfig, ModelConfig
    from lineartgen.models import create_model
    from lineartgen.models.serialization import save_inference

    config = ExperimentConfig(
        model=ModelConfig(architecture="nested", channels=(2, 3), nested_channels=((4, 5),))
    )
    model = create_model(config.model, rngs=nnx.Rngs(0))
    save_inference(model, config, tmp_path / "model", provenance="count fixture")
    weights = tmp_path / "model/weights.npz"
    for i in (0, 1):
        assert analysis.parameter_count(weights, index=i) == sum(
            x.size for x in jax.tree.leaves(nnx.state(model.extract(i), nnx.Param))
        )
    assert analysis.parameter_count(weights) == analysis.parameter_count(weights, index=1)
    with np.load(weights) as data:
        assert all(isinstance(json.loads(k), list) for k in data.files)


def test_size_selection_rejects_regression_and_final_does_not_reselect(tmp_path, monkeypatch):
    analysis = module("analyze_completion_campaign", monkeypatch)
    protocol = analysis.read(Path(__file__).parents[1] / "configs/completion-protocol.json")
    protocol["bootstrap_draws"] = 100
    analysis.write(tmp_path / "protocol.json", protocol)
    analysis.write(tmp_path / "decisions/budget.json", dict(step=10000))
    cases = [
        dict(kind="real", source="a"),
        dict(kind="straight", variant=1),
        dict(kind="real", source="a"),
        dict(kind="real", source="a"),
        dict(kind="straight", variant=1),
    ]
    manifest = tmp_path / "development/cases.json"
    analysis.write(manifest, dict(cases=cases))
    groups = [
        dict(suite="matched_real"),
        dict(suite="geometry"),
        dict(suite="sketch_real", family="blur", severity="strong"),
        dict(suite="sketch_real", family="strokes", severity="strong"),
        dict(suite="sketch_geometry", family="strokes", severity="strong"),
    ]
    cohorts = {s: {r: s + r for r in ("shared", "core", "baseline")} for s in ("42", "43", "44")}
    analysis.write(tmp_path / "cohorts.json", cohorts)
    for cohort in cohorts.values():
        for role, name in cohort.items():
            arm = tmp_path / "arms" / name
            analysis.write(
                arm / "config.json",
                dict(model=dict(nested_channels=[[], []] if role == "shared" else [])),
            )
            weights = arm / "snapshots/step-10000/model/ema/weights.npz"
            weights.parent.mkdir(parents=True)
            arrays = (
                {json.dumps(["heads", i, "bias"]): np.zeros(i + 2) for i in range(3)}
                if role == "shared"
                else {json.dumps(["head", "bias"]): np.zeros(2 if role == "core" else 40)}
            )
            np.savez(weights, **arrays)
            for mode in ("raw", "ema"):
                rows = []
                for subnet in range(3 if role == "shared" else 1):
                    for i, group in enumerate(groups):
                        rows.append(
                            dict(
                                group,
                                case=i,
                                method="model",
                                subnet=subnet,
                                strength=0.5,
                                denoise_steps=1,
                                recursion_levels=3,
                                protected_max_error=0.0,
                                active_target_ink_fraction=0.1,
                                active_ink_f1=0.93 if role == "shared" and subnet == 0 else 0.96,
                                profile_width_mae_px=0.01,
                                profile_missing_fraction=0.01,
                                connection_cut_rate=0.01,
                                connection_bridge_rate=0.01,
                            )
                        )
                analysis.write(
                    arm / f"evaluation/step-10000-{mode}/metrics.json",
                    dict(
                        checkpoint_step=10000,
                        ema=mode == "ema",
                        cases=5,
                        manifest_sha256=analysis.sha(manifest),
                        noise_generator="numpy",
                        recursion="runtime",
                        results=rows,
                    ),
                )
    result = analysis.analyze(tmp_path)
    assert result["smallest_eligible"] == "independent_s"
    assert not result["candidate_decisions"][0]["eligible"]
    assert all(
        v["kind"] == "noninferiority" for v in result["candidate_decisions"][0]["violations"]
    )
    analysis.write(
        tmp_path / "selection-seal.json", dict(artifacts={}, selected_model="independent_s")
    )
    shutil.copytree(tmp_path / "development", tmp_path / "final")
    for cohort in cohorts.values():
        for role, name in cohort.items():
            arm = tmp_path / "arms" / name
            shutil.copytree(arm / "evaluation", arm / "final-evaluation")
            if role == "core":
                path = arm / "final-evaluation/step-10000-ema/metrics.json"
                bad = analysis.read(path)
                bad["results"][-1]["active_ink_f1"] = 0.75
                analysis.write(path, bad)
    result = analysis.analyze(tmp_path, final=True)
    assert result["sealed_selection"] == "independent_s"
    assert result["smallest_eligible"] is None and not result["selection_confirmed"]
    independent = next(d for d in result["candidate_decisions"] if d["model"] == "independent_s")
    assert any(v["kind"] == "absolute" for v in independent["violations"])


def test_training_audit_detects_gaps_and_wrong_generated_seed(tmp_path, monkeypatch):
    audit = module("audit_completion_campaign", monkeypatch)
    audit.write(tmp_path / "training/progress.json", dict(step=120, runs=["model"]))
    audit.write(
        tmp_path / "config.json",
        dict(train=dict(seed=47, batch_size=1, accumulate_steps=2, log_every=10)),
    )
    rows = [
        dict(
            step=i,
            scale_level=(i - 1) % 6,
            recursion_levels=(i - 1) % 6 + 1,
            strengths=dict(model=[0.0, 0.5]),
            mask_areas=dict(model=[0, 100]),
            generated=dict(model=[dict(item=1, key=[47, i - 1, 1, 0])]),
            updates=dict(model=0.5),
        )
        for i in range(1, 121)
    ]
    path = tmp_path / "training/timings.jsonl"

    def write_rows():
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    write_rows()
    metrics = tmp_path / "training/model/metrics.jsonl"
    metrics.parent.mkdir()
    metrics.write_text(
        "".join(
            json.dumps(dict(step=i, phase="train", loss=0.1)) + "\n" for i in range(10, 121, 10)
        )
    )
    result = audit.training_audit(tmp_path)
    assert result["generated_fraction"] == result["identity_fraction"] == 0.5
    assert result["update_seconds"] == 60
    rows[0]["generated"]["model"][0]["key"][0] = 48
    write_rows()
    with pytest.raises(AssertionError):
        audit.training_audit(tmp_path)
    rows.pop(0)
    write_rows()
    with pytest.raises(AssertionError):
        audit.training_audit(tmp_path)


def test_supplementary_diagnostics_require_each_iteration_and_correct_depth(tmp_path, monkeypatch):
    module_ = module("supplement_campaign_diagnostics", monkeypatch)
    from lineartgen.config import ExperimentConfig, save_config

    cases, arm = tmp_path / "cases", tmp_path / "arm"
    module_.write(cases / "cases.json", dict(cases=[dict(scale_level=0), dict(scale_level=5)]))
    arm.mkdir()
    save_config(ExperimentConfig(), arm / "config.json")
    rows = [
        dict(
            case=i,
            subnet=0,
            denoise_steps=steps,
            method="model",
            recursion_levels=1 if i == 0 else 6,
            protected_max_error=0.0,
        )
        for i in range(2)
        for steps in (2, 5)
    ]
    result = dict(
        results=rows,
        cases=2,
        denoise_steps=[2, 5],
        manifest_sha256=module_.sha(cases / "cases.json"),
        checkpoint_step=10000,
        ema=True,
        noise_generator="numpy",
        recursion="dataset",
    )
    path = tmp_path / "metrics.json"
    module_.write(path, result)
    module_.validate(path, arm, cases, 10000, "ema", (2, 5), "dataset")
    rows[0]["recursion_levels"] = 3
    module_.write(path, result)
    with pytest.raises(AssertionError):
        module_.validate(path, arm, cases, 10000, "ema", (2, 5), "dataset")
    rows[0]["recursion_levels"] = 1
    rows.pop()
    module_.write(path, result)
    with pytest.raises(AssertionError):
        module_.validate(path, arm, cases, 10000, "ema", (2, 5), "dataset")
    with pytest.raises(AssertionError, match="locked campaign controller"):
        module_.inherited_lock(tmp_path)
    with (tmp_path / "controller.lock").open("a+") as lock:
        assert module_.inherited_lock(tmp_path) == (lock.fileno(),)
