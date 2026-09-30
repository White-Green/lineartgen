"""Make the final descriptive report from audited, sealed campaign artifacts."""

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate(rows, keys, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for identity, values in sorted(groups.items()):
        item = dict(zip(keys, identity, strict=True))
        item["seeds"] = [row["seed"] for row in values]
        for metric in metrics:
            observed = [row[metric] for row in values if row.get(metric) is not None]
            item[metric] = (
                dict(mean=statistics.mean(observed), minimum=min(observed), maximum=max(observed))
                if observed
                else None
            )
        output.append(item)
    return output


def summarize(root):
    names = [
        "protocol.json",
        "cohorts.json",
        "selection-seal.json",
        "development-summary.json",
        "final-summary.json",
        "diagnostic-summary.json",
        "model-verification.json",
        "runtime-verification.json",
        "blind-observations.json",
        "blind-review.json",
        "blind/manifest.json",
        "integrity.json",
        "cache-integrity.json",
        "evidence-sha256.json",
        "training-source.json",
        "final-stage-source-v5.json",
        "cache-migration-source.json",
    ]
    sources = {name: sha(root / name) for name in names}
    integrity, cache = read(root / "integrity.json"), read(root / "cache-integrity.json")
    assert integrity["paired_training_streams_exact"] and integrity["final_sources_disjoint"]
    assert integrity["source_and_input_hashes_verified"] and cache["cache_payload_hashes_verified"]
    seal = read(root / "selection-seal.json")
    stages = {}
    metrics = [
        "active_ink_f1",
        "profile_width_mae_px",
        "profile_missing_fraction",
        "connection_cut_rate",
        "connection_bridge_rate",
    ]
    for stage in ("development", "final"):
        data = read(root / f"{stage}-summary.json")
        stages[stage] = dict(
            aggregate=aggregate(data["per_seed_summary"], ["model", "mode", "subset"], metrics),
            per_seed=data["per_seed_summary"],
            decisions=data["candidate_decisions"],
            real_ink_aggregate=aggregate(
                data["real_ink_diagnostics"],
                ["model", "mode", "subset", "population"],
                ["active_ink_f1", "active_mae", "active_background_mae"],
            ),
            real_ink_per_seed=data["real_ink_diagnostics"],
            real_ink_definition=data["real_ink_definition"],
            smallest_eligible=data["smallest_eligible"],
        )
    models = read(root / "model-verification.json")
    runtime = read(root / "runtime-verification.json")
    observations, review = read(root / "blind-observations.json"), read(root / "blind-review.json")
    assert review["observations_sha256"] == sources["blind-observations.json"]
    assert observations["recorded_utc"] < review["opened_utc"]
    assert not observations["key_opened_before_recording"] and len(observations["panels"]) == 32
    assert len(runtime["records"]) == len(models["records"]) == 18
    assert all(
        row[key]
        for record in runtime["records"]
        for row in record["bgra"]
        for key in ("identity_exact", "protected_exact", "deterministic")
    )
    diagnostics = read(root / "diagnostic-summary.json")
    return dict(
        root=str(root.resolve()),
        source_sha256=sha(Path(__file__)),
        artifacts=sources,
        conclusion="All planned experiments finished; no eligible model and no minimum "
        "sufficient size established.",
        selected_model=seal["selected_model"],
        selection_created_utc=seal["created_utc"],
        step=seal["step"],
        final_selection_confirmed=read(root / "final-summary.json")["selection_confirmed"],
        aggregation="Arithmetic mean and min/max over three fitted model/data seeds; ranges "
        "are not confidence intervals. Paired hierarchical intervals are preserved in "
        "decisions. No final-test reselection.",
        stages=stages,
        training_audit=integrity,
        cache_audit=cache,
        timings=models["timings"],
        timing_devices=models["devices"],
        prefix_max_error=max(c["error"] for r in models["records"] for c in r["comparisons"]),
        onnx_exports=len(runtime["records"]),
        bgra_cases=sum(len(r["bgra"]) for r in runtime["records"]),
        onnx_max_error=max(r["runtime_max_error"] for r in runtime["records"]),
        export_scope=models["export_scope"],
        diagnostic_scope={k: diagnostics[k] for k in ("selection_policy", "limitations")},
        diagnostic_groups=dict(Counter(r["label"] for r in diagnostics["records"])),
        diagnostics=diagnostics["records"],
        blind_observations=observations,
        blind_review=review,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
