"""Audit all committed updates, paired data streams, fixed grids and sealed final evidence."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from analyze_completion_campaign import check_seal
from campaign_protocol import needs_continuation, read, report, sha, write
from supplement_campaign_diagnostics import validate as validate_diagnostic


def training_audit(arm):
    progress = read(arm / "training/progress.json")
    config = read(arm / "config.json")
    assert progress["runs"] == ["model"]
    step = progress["step"]
    digest = hashlib.sha256()
    scales, depths, updates = {}, {}, []
    generated, strengths, masks = 0, [], []
    with (arm / "training/timings.jsonl").open() as stream:
        for expected, line in enumerate(stream, 1):
            row = json.loads(line)
            assert row["step"] == expected
            paired = {k: row[k] for k in ("step", "scale_level", "recursion_levels")}
            for field in ("strengths", "mask_areas", "generated"):
                assert set(row[field]) == {"model"}
                paired[field] = row[field]["model"]
            batch = config["train"]["batch_size"] * config["train"]["accumulate_steps"]
            assert len(paired["strengths"]) == len(paired["mask_areas"]) == batch
            for item in paired["generated"]:
                assert item["key"][:3] == [config["train"]["seed"], expected - 1, item["item"]]
            digest.update(json.dumps(paired, sort_keys=True).encode())
            digest.update(b"\n")
            scales[row["scale_level"]] = scales.get(row["scale_level"], 0) + 1
            depths[row["recursion_levels"]] = depths.get(row["recursion_levels"], 0) + 1
            assert np.isfinite(row["updates"]["model"])
            updates.append(row["updates"]["model"])
            generated += len(paired["generated"])
            strengths.extend(paired["strengths"])
            masks.extend(paired["mask_areas"])
    assert len(updates) == step
    assert set(scales) == set(range(6)) and set(depths) == set(range(1, 7))
    metrics = 0
    with (arm / "training/model/metrics.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            assert 0 < row["step"] <= step
            assert all(np.isfinite(v) for v in row.values() if isinstance(v, (int, float)))
            metrics += row["phase"] == "train"
    assert metrics == step // config["train"]["log_every"]
    return dict(
        step=step,
        paired_data_sha256=digest.hexdigest(),
        scales=scales,
        recursion_depths=depths,
        generated_examples=generated,
        generated_fraction=generated / (step * batch),
        identity_fraction=float(np.mean(np.array(strengths) == 0)),
        mean_strength=float(np.mean(strengths)),
        mean_mask_area=float(np.mean(masks)),
        steady_update_median_seconds=float(np.median(updates[100:])),
        update_seconds=float(sum(updates)),
        timing_scope="Actual optimizer updates; median excludes first 100 for compilation. "
        "Includes gradient accumulation, excludes data preparation and evaluation.",
    )


def audit(root):
    seal = check_seal(root)
    protocol, cohorts = read(root / "protocol.json"), read(root / "cohorts.json")
    frozen = read(root / "training-source.json")
    assert sha(frozen["archive"]) == frozen["archive_sha256"]
    for path, digest in frozen["file_sha256"].items():
        assert sha(path) == digest, path
    for path, digest in read(root / "dependency-source.json")["file_sha256"].items():
        assert sha(path) == digest, path
    for name, digest in read(root / "development-input-sha256.json").items():
        assert sha(root / "development" / name) == digest
    for name, digest in read(root / "final/input-sha256.json").items():
        assert sha(root / "final" / name) == digest
    final_manifest = read(root / "final/cases.json")
    assert final_manifest["selection_seal_sha256"] == sha(root / "selection-seal.json")
    assert len(final_manifest["cases"]) == 1632
    dataset = final_manifest["dataset"]
    groups = [set(dataset[k]) for k in ("train", "validation", "final_test")]
    assert [len(g) for g in groups] == [891, 64, 64]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    held = {t["source"] for t in final_manifest["targets"] if t["kind"] == "real"}
    assert held == groups[2]
    audits, total_predictions = {}, 0
    for arm in sorted((root / "arms").iterdir()):
        if not (arm / "training/progress.json").exists():
            continue
        audits[arm.name] = training_audit(arm)
        assert read(arm / "training/model/dataset.json") == dataset
        for scope, cases in (("evaluation", "development"), ("final-evaluation", "final")):
            for path in sorted((arm / scope).glob("step-*/metrics.json")):
                _, step, mode = path.parent.name.split("-")
                result = report(arm, int(step), mode, root / cases, scope=scope)
                total_predictions += len(result["results"])
        for path in sorted((arm / "diagnostics").glob("*/metrics.json")):
            result = read(path)
            validate_diagnostic(
                path,
                arm,
                root / "development",
                result["checkpoint_step"],
                "ema" if result["ema"] else "raw",
                tuple(result["denoise_steps"]),
                result["recursion"],
            )
            total_predictions += len(result["results"])
    for seed, cohort in cohorts.items():
        expected = audits[cohort["shared"]]["paired_data_sha256"]
        for name in cohort.values():
            assert audits[name]["step"] == seal["step"]
            assert audits[name]["paired_data_sha256"] == expected, (seed, name)
        initialized = read(root / "initialization" / f"{cohort['core']}.json")
        assert initialized["exact"]
        for path, digest in initialized["files"].items():
            assert sha(path) == digest
        for name in cohort.values():
            arm = root / "arms" / name
            for mode in ("raw", "ema"):
                for step in protocol["long_steps"]:
                    if step <= seal["step"]:
                        report(arm, step, mode, root / "development")
                report(arm, seal["step"], mode, root / "final", scope="final-evaluation")
    model_check, runtime = (
        read(root / "model-verification.json"),
        read(root / "runtime-verification.json"),
    )
    expected = {
        (name, mode)
        for cohort in cohorts.values()
        for name in cohort.values()
        for mode in ("raw", "ema")
    }
    for verification in (model_check, runtime):
        actual = [(r["arm"], r["mode"]) for r in verification["records"]]
        assert len(actual) == len(expected) and set(actual) == expected
    for row in runtime["records"]:
        assert sha(row["onnx"]) == row["sha256"] and len(row["bgra"]) == 4
    blind = read(root / "blind/manifest.json")
    assert len(blind["panels"]) == 32
    for panel in blind["panels"]:
        assert sha(root / "blind" / panel["image"]) == panel["sha256"]
        assert sha(root / "blind" / panel["zoom"]) == panel["zoom_sha256"]
    ceiling = None
    if seal["step"] == 100000:
        ceiling = needs_continuation(root, list(cohorts["42"].values()), 60000, 100000, protocol)
    write(
        root / "integrity.json",
        dict(
            training=audits,
            fixed_predictions=total_predictions,
            final_predictions=1632 * 5 * 3 * 2,
            paired_training_streams_exact=True,
            final_sources_disjoint=True,
            source_and_input_hashes_verified=True,
            ceiling_convergence=ceiling,
            final_blank_real_targets=sum(
                t["kind"] == "real" and t["ink_fraction"] < 0.001 for t in final_manifest["targets"]
            ),
            selected_model=seal["selected_model"],
            final_selection_confirmed=read(root / "final-summary.json")["selection_confirmed"],
            note="Numerical integrity only; qualitative review and final report are separate.",
            source_sha256=sha(__file__),
        ),
    )
    artifacts = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.name not in ("evidence-sha256.json", "controller.lock", "continuation.json")
        and p.suffix not in (".log", ".partial")
        and "events" not in p.parts
    ]
    write(root / "evidence-sha256.json", {str(p): sha(p) for p in sorted(artifacts)})
    return root / "integrity.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(audit(args.root.resolve()))


if __name__ == "__main__":
    main()
