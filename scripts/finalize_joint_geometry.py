"""Finalize matched fresh full width/merge and mixed-opacity geometry training."""

import argparse
import json
from pathlib import Path

import numpy as np
from campaign_protocol import read, sha, write
from continuous_width_data import width_choice
from joint_geometry_study import (
    CACHE_SOURCE,
    PARENT,
    REPORT_METHODS,
    SAVED,
    SEED,
    STEPS,
    check,
    select,
)

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
root = p.parse_args().root.resolve()
check(root)
rows = [json.loads(x) for x in (root / "training/sampling.jsonl").read_text().splitlines()]
controls = [json.loads(x) for x in (PARENT / "training/sampling.jsonl").read_text().splitlines()]
assert len(rows) == len(controls) == STEPS
eligible = 0
selected = {m: 0 for m in REPORT_METHODS}
composed = 0
opacity_ratios = []
type_pairs = set()
records = read(root / "data.json")["records"]
changed = 0
for step, (row, c) in enumerate(zip(rows, controls, strict=True)):
    assert row["step"] == c["step"] == step + 1
    selection = select(step)
    for k, v in selection.items():
        assert row[k] == v
        if k != "overlays":
            assert c[k] == v
    widths = [width_choice(SEED, step, item, i) for item, i in enumerate(selection["indices"])]
    assert row["source_widths"] == widths
    flags = [x["enabled"] for x in selection["overlays"]]
    same = row["input_sha256"]["mixed"] == row["input_sha256"]["merge"]
    if not any(flags):
        assert same
        assert row["shared_target_sha256"]["mixed"] == row["shared_target_sha256"]["merge"]
        assert row["candidate_payload_sha256"]["mixed"] == row["candidate_payload_sha256"]["merge"]
    changed += not same
    composed += sum(flags)
    for i, opts, overlay in zip(
        selection["indices"], selection["augmentation"], selection["overlays"], strict=True
    ):
        if overlay["enabled"]:
            opacity_ratios.append(overlay["augmentation"]["opacity"] / opts["opacity"])
            type_pairs.add((records[i]["kind"], records[overlay["index"]]["kind"]))
    eligible += sum(row["merge_eligible"])
    for method in REPORT_METHODS:
        selected[method] += row["metrics"][method]["merged_fraction"] * 4
assert read(root / "prefix-audit.json")["passed"]
assert read(root / "initial-audit.json")["six_depth_functions_exact"]
audit = read(root / "snapshot-audit.json")
assert audit["raw_ema_exact"] and audit["exports"] == 24
for f, h in audit["bindings"].items():
    assert sha(Path(f)) == h
for step in SAVED[1:]:
    assert read(root / f"training/restore-{step}.json")["full_state_exact"]
for until in (4500, 9000, 13500, 18000):
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
    assert len(m["rows"]) == 4608 and m["protected_exact"] and m["coverage_exact"]
    bindings = (
        read(root / group / "prediction-binding.json")
        if group == "evaluation"
        else read(root / group / "binding.json")["predictions"]
    )
    for f, h in bindings.items():
        assert sha(root / group / f) == h
    reused = 0
    for file in (root / group / "predictions").glob("*.npz"):
        with np.load(file) as a, np.load(CACHE_SOURCE / group / "predictions" / file.name) as b:
            np.testing.assert_array_equal(a["initial"], b["initial"])
            reused += 1
    reuse[group] = reused
sweep = read(root / "width-sweep/metrics.json")
assert sweep["outputs"] == 108 and sweep["states_unchanged"]
for f, h in sweep["bindings"].items():
    assert sha(Path(f)) == h
scene = read(root / "scene512/metrics.json")
assert scene["model_outputs"] == 192 and scene["protected_exact"]
for f, h in scene["bindings"].items():
    assert sha(Path(f)) == h
for file in (root / "scene512").glob("predictions-*.npz"):
    with np.load(file) as a, np.load(CACHE_SOURCE / "scene512" / file.name) as b:
        for k in ("initial_float", "initial_uint8"):
            np.testing.assert_array_equal(a[k], b[k])
write(
    root / "diagnostics/integrity.json",
    dict(
        uncomposed_batches_exact=True,
        merge_eligibility_exact=True,
        composed_examples=composed,
        opacity_ratio_min=float(min(opacity_ratios)),
        opacity_ratio_max=float(max(opacity_ratios)),
        type_pairs=sorted(type_pairs),
        changed_input_batches=changed,
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
        stage="complete_joint_geometry",
        steps=STEPS,
        trained_arms=2,
        reused_control=False,
        outputs=9516,
        native_pages_reviewed=34,
        adoption=False,
        fresh_confirmation=False,
        evidence={f: sha(root / f) for f in files},
    ),
)
print(dict(complete=True, integrity=read(root / "diagnostics/integrity.json")), flush=True)
