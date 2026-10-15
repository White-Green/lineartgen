"""Finalize forced family selection and the reused EY L4 control."""

import argparse
import json
from pathlib import Path

import numpy as np
from campaign_protocol import read, sha, write
from forced_geometry_study import (
    CONTROL,
    PARENT,
    REPORT_METHODS,
    SAVED,
    STEPS,
    SUPPORT,
    check,
    select,
)

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
root = p.parse_args().root.resolve()
check(root)
rows = [json.loads(x) for x in (root / "training/sampling.jsonl").read_text().splitlines()]
assert len(rows) == STEPS
eligible = {m: 0 for m in REPORT_METHODS}
selected = {m: 0 for m in REPORT_METHODS}
dropped_edges = 0
total_edges = 0
control_rows = [
    json.loads(x) for x in (CONTROL / "training/sampling.jsonl").read_text().splitlines()
]
support_rows = [
    json.loads(x) for x in (SUPPORT / "training/sampling.jsonl").read_text().splitlines()
]
for step, row in enumerate(rows):
    assert row["step"] == step + 1
    selection = select(step)
    for k, v in selection.items():
        assert row[k] == v
    old = control_rows[step]
    for key in ("input_sha256", "shared_target_sha256"):
        assert (
            row[key]["control"]
            == row[key]["forced"]
            == row[key]["supported"]
            == old[key]["quartic"]
        )
    assert row["candidate_payload_sha256"]["forced"] == old["candidate_payload_sha256"]["quartic"]
    assert row["metrics"]["control"] == old["metrics"]["quartic"]
    flags = row["merge_eligible"]
    assert flags["control"] == flags["forced"] == old["merge_eligible"]["quartic"]
    assert all(not a or b for a, b in zip(flags["forced"], flags["supported"], strict=True))
    for method in ("forced", "supported"):
        assert row["metrics"][method]["merged_fraction"] == sum(flags[method]) / 4
    prior = support_rows[step]
    assert (
        row["candidate_payload_sha256"]["supported"] == prior["candidate_payload_sha256"]["support"]
    )
    assert flags["supported"] == prior["merge_eligible"]["support"]
    assert row["context_keep"] == old["context_keep"]
    edges = np.asarray(row["context_keep"])[1:]
    dropped_edges += int(np.sum(edges == 0))
    total_edges += edges.size
    for method in REPORT_METHODS:
        eligible[method] += sum(flags[method])
        selected[method] += row["metrics"][method]["merged_fraction"] * 4
assert eligible == dict(control=1207, forced=1207, supported=1320)
assert selected["forced"] == 1207 and selected["supported"] == 1320
assert read(root / "prefix-audit.json")["passed"]
assert read(root / "initial-audit.json")["six_depth_diagnostic_functions_exact"]
audit = read(root / "snapshot-audit.json")
assert audit["raw_ema_exact"] and audit["exports"] == 32
for f, h in audit["bindings"].items():
    assert sha(Path(f)) == h
for step in SAVED[1:]:
    assert read(root / f"training/restore-{step}.json")["full_state_exact"]
for until in SAVED[2:]:
    assert read(root / f"driver-train-{until}.json")["returncode"] == 0
for phase in ("evaluation", "stress", "scene512", "width-sweep"):
    assert read(root / f"driver-{phase}.json")["returncode"] == 0
tests = read(root / "test-results.json")
assert tests["passed"]
assert read(root / "preflight.json")["passed"]
for f, h in tests["logs"].items():
    assert sha(root / f) == h
notes = read(root / "diagnostics/review-observations.json")
assert len(notes) == 34
for group, folder, count in [
    ("evaluation", "evaluation/native-review", 12),
    ("stress", "stress/native-review", 10),
    ("scene512", "scene512", 6),
    ("width-sweep", "width-sweep", 6),
]:
    records = read(root / folder / "manifest.json")["records"]
    assert len(records) == count
    for r in records:
        assert sha(root / folder / r["file"]) == r["sha256"]
    write(
        root / group / "visual-review.json",
        dict(
            reviewed=True,
            native_scale=True,
            files={r["file"]: r["sha256"] for r in records},
            observations={r["file"]: notes[folder + "/" + r["file"]] for r in records},
        ),
    )
reuse = {}
for group in ("evaluation", "stress"):
    m = read(root / group / "metrics.json")
    assert len(m["rows"]) == 6144 and m["protected_exact"] and m["coverage_exact"]
    bindings = (
        read(root / group / "prediction-binding.json")
        if group == "evaluation"
        else read(root / group / "binding.json")["predictions"]
    )
    for f, h in bindings.items():
        assert sha(root / group / f) == h
    reused = 0
    for file in (root / group / "predictions").glob("*.npz"):
        with np.load(file) as a, np.load(CONTROL / group / "predictions" / file.name) as b:
            for m in ("initial", "control"):
                np.testing.assert_array_equal(a[m], b["quartic" if m == "control" else m])
            reused += 1
    reuse[group] = reused
sweep = read(root / "width-sweep/metrics.json")
assert sweep["outputs"] == 144 and sweep["states_unchanged"]
for f, h in sweep["bindings"].items():
    assert sha(Path(f)) == h
for file in (root / "width-sweep").glob("wide-*.npz"):
    with np.load(file) as a, np.load(CONTROL / "width-sweep" / file.name) as b:
        for k in ("source", "initial", "control"):
            np.testing.assert_array_equal(a[k], b[k.replace("control", "quartic")])
scene = read(root / "scene512/metrics.json")
assert scene["model_outputs"] == 256 and scene["protected_exact"]
for f, h in scene["bindings"].items():
    assert sha(Path(f)) == h
for file in (root / "scene512").glob("predictions-*.npz"):
    with np.load(file) as a, np.load(CONTROL / "scene512" / file.name) as b:
        for k in ("initial_float", "initial_uint8", "control_float", "control_uint8"):
            np.testing.assert_array_equal(a[k], b[k.replace("control", "quartic")])
write(
    root / "diagnostics/integrity.json",
    dict(
        all_training_inputs_original_targets_masks_exact=True,
        capped_candidate_payloads_exact=True,
        dropped_edges=dropped_edges,
        total_edges=total_edges,
        reused_control=True,
        eligible_examples=eligible,
        selected_merged_examples=selected,
        reused_prediction_files_exact=reuse,
        last600_update_seconds={
            m: float(np.mean([r["seconds"][m] for r in rows[-600:]])) for m in REPORT_METHODS
        },
        quantization_output_rms={
            m: dict(
                mean=float(np.mean([r["quantization_output_rms"][m] for r in scene["rows"]])),
                maximum=float(max(r["quantization_output_rms"][m] for r in scene["rows"])),
            )
            for m in REPORT_METHODS
        },
    ),
)
assert sha(root / "fields.npy") == sha(PARENT / "fields.npy")
files = [
    "plan.json",
    "source.json",
    "test-results.json",
    "prefix-audit.json",
    "snapshot-audit.json",
    "training/sampling.jsonl",
    "diagnostics/review-observations.json",
    "diagnostics/integrity.json",
    "driver-evaluation.json",
    "diagnostics/decision.json",
    "evaluation/metrics.json",
    "stress/metrics.json",
    "scene512/metrics.json",
    "width-sweep/metrics.json",
    "width-sweep/visual-review.json",
]
write(
    root / "completion.json",
    dict(
        stage="complete_forced_geometry",
        steps=STEPS,
        trained_arms=2,
        reused_control=True,
        outputs=12688,
        native_pages_reviewed=34,
        adoption=False,
        fresh_confirmation=False,
        evidence={f: sha(root / f) for f in files},
    ),
)
print(dict(complete=True, integrity=read(root / "diagnostics/integrity.json")), flush=True)
