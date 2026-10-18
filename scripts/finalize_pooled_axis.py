"""Verify actual training, matched-control predictions and all fixed native reviews."""

import argparse
import json
from pathlib import Path

import numpy as np
from campaign_protocol import read, sha, write
from pooled_axis_study import HH, HJ, METHODS, SAVED, STEPS, check

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
root = parser.parse_args().root.resolve()
check(root)
rows = [json.loads(s) for s in (root / "training/sampling.jsonl").read_text().splitlines()]
old = [json.loads(s) for s in (HH / "training/sampling.jsonl").read_text().splitlines()][:STEPS]
assert len(rows) == len(old) == STEPS
families = {}
whole = promoted = 0
reference_errors = []
reference_nonexact = 0


def reference_compare(actual, expected):
    global reference_nonexact
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    reference_errors.append(float(np.max(np.abs(actual - expected))))
    reference_nonexact += int(not np.array_equal(actual, expected))
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


for k, (a, b) in enumerate(zip(rows, old, strict=True)):
    assert a["step"] == k + 1
    for key in (
        "input_sha256",
        "candidate_payload_sha256",
        "context_keep",
        "audit",
        "coherent_extension",
        "consistent_original",
    ):
        assert a[key] == b[key], (k, key)
    family = a["audit"]["family"]
    families[family] = families.get(family, 0) + 1
    if family == "ambiguous":
        whole += sum(a["audit"]["alternate_allowed"])
    promoted += sum(a["coherent_extension"]["promoted"])
    for m in METHODS:
        assert all(np.isfinite(v) for v in a["metrics"][m].values())
        bits = int(a["metrics"][m]["selected_alternate_bits"])
        assert [(bits >> j) & 1 for j in range(4)] == [int(not v) for v in a["consistent_original"]]
assert families == dict(legacy=750, long_bar_fill=300, ambiguous=150)
expected = read(root / "plan.json")["expected_counts"]
assert whole == expected["whole"] and promoted == expected["promoted"]
assert (
    sum(sum(a["coherent_extension"]["promoted"]) for a in rows if a["audit"]["family"] == "legacy")
    == expected["legacy_promoted"]
)
assert not read(root / "diagnostics/decision.json")["adoption"]
for name in ("test-results", "preflight", "prefix-audit"):
    assert read(root / f"{name}.json")["passed"]
snap = read(root / "snapshot-audit.json")
assert snap["raw_ema_exact"] and snap["exports"] == 2 * len(METHODS) * len(SAVED)
for f, h in snap["bindings"].items():
    assert sha(Path(f)) == h
for step in SAVED[1:]:
    assert read(root / f"training/restore-{step}.json")["full_state_exact"]
phases = [
    "preflight",
    "train-1200",
    "evaluation",
    "stress",
    "scene512",
    "width-sweep",
    "ambiguity",
    "geometry",
    "collision",
    "details",
    "orientation",
    "mask",
    "coherent",
]
for phase in phases:
    assert read(root / f"driver-{phase}.json")["returncode"] == 0
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
        assert notes[folder + "/" + r["file"]] and sha(root / folder / r["file"]) == r["sha256"]
        for f, h in r.get("bindings", {}).items():
            assert sha(Path(f)) == h
    write(
        root / group / "visual-review.json",
        dict(
            reviewed=True,
            native_scale=True,
            files={r["file"]: r["sha256"] for r in records},
            observations={r["file"]: notes[folder + "/" + r["file"]] for r in records},
        ),
    )
counts = {}
for group in ("evaluation", "stress"):
    metrics = read(root / group / "metrics.json")
    assert len(metrics["rows"]) == 4608 and metrics["protected_exact"] and metrics["coverage_exact"]
    bindings = (
        read(root / group / "prediction-binding.json")
        if group == "evaluation"
        else read(root / group / "binding.json")["predictions"]
    )
    for f, h in bindings.items():
        assert sha(root / group / f) == h
    count = 0
    for file in (root / group / "predictions").glob("*.npz"):
        with np.load(file) as a, np.load(HJ / group / "predictions" / file.name) as b:
            for method, old_method in (("initial", "consistent"),):
                reference_compare(a[method], b[old_method])
        count += 1
    assert count == 1536
    counts[group] = count
for group, key, count in [("scene512", "model_outputs", 192), ("width-sweep", "outputs", 108)]:
    metrics = read(root / group / "metrics.json")
    assert metrics[key] == count
    for f, h in metrics["bindings"].items():
        assert sha(Path(f)) == h
assert read(root / "scene512/metrics.json")["protected_exact"]
assert read(root / "width-sweep/metrics.json")["states_unchanged"]
for file in (root / "scene512").glob("predictions-*.npz"):
    with np.load(file) as a, np.load(HJ / "scene512" / file.name) as b:
        for method, old_method in (("initial", "consistent"),):
            for encoding in ("float", "uint8"):
                reference_compare(a[method + "_" + encoding], b[old_method + "_" + encoding])
for file in (root / "width-sweep").glob("wide-*.npz"):
    with np.load(file) as a, np.load(HJ / "width-sweep" / file.name) as b:
        for key, old_key in (
            ("source", "source"),
            ("initial", "consistent"),
        ):
            if key == "source":
                np.testing.assert_array_equal(a[key], b[old_key])
            else:
                reference_compare(a[key], b[old_key])
for group, outputs, pages in [("ambiguity", 12, 3), ("geometry", 144, 9), ("collision", 18, 3)]:
    folder = root / "diagnostics" / group
    metrics = read(folder / "metrics.json")
    assert metrics["outputs"] == outputs and metrics["states_unchanged"]
    if group == "geometry":
        assert (
            metrics["primary_outputs"] == 72
            and metrics["repeated_anchors_within_tolerance"]
            and len(metrics["comparisons"]) == 72
        )
    for f, h in metrics["bindings"].items():
        assert sha(Path(f)) == h
    records = read(folder / "manifest.json")["records"]
    observations = read(folder / "observations.json")
    assert len(records) == len(observations) == pages
    for r in records:
        assert observations[r["file"]] and sha(folder / r["file"]) == r["sha256"]
    if group in ("ambiguity", "geometry", "collision"):
        for file in folder.glob("*.npz"):
            with np.load(file) as a, np.load(HJ / "diagnostics" / group / file.name) as b:
                if "initial" in a.files:
                    reference_compare(a["initial"], b["consistent"])
details = root / "diagnostics/details"
records = read(details / "manifest.json")["records"]
observations = read(details / "observations.json")
assert len(records) == len(observations) == 5
for r in records:
    assert observations[r["file"]] and sha(details / r["file"]) == r["sha256"]
    for f, h in r["bindings"].items():
        assert sha(Path(f)) == h
orientation = root / "diagnostics/orientation"
om = read(orientation / "metrics.json")
assert om["outputs"] == 32 and om["states_unchanged"]
for f, h in om["bindings"].items():
    assert sha(Path(f)) == h
records = read(orientation / "manifest.json")["records"]
notes = read(orientation / "observations.json")
assert len(records) == len(notes) == 4
for r in records:
    assert notes[r["file"]] and sha(orientation / r["file"]) == r["sha256"]
for folder, count in (("support-preview", 3), ("mask-probes", 6), ("coherent-probes", 12)):
    manifest = read(root / folder / "manifest.json")
    notes = read(root / folder / "observations.json")
    assert len(manifest["records"]) == len(notes) == count
    for r in manifest["records"]:
        assert notes[r["file"]] and sha(root / folder / r["file"]) == r["sha256"]
    for p, h in manifest.get("bindings", {}).items():
        assert sha(Path(p)) == h
mask = read(root / "mask-probes/metrics.json")
assert mask["outputs"] == 36 and mask["states_unchanged"] and mask["protected_exact"]
for p, h in mask["bindings"].items():
    assert sha(Path(p)) == h
coherent = read(root / "coherent-probes/metrics.json")
assert coherent["outputs"] == 96 and coherent["states_unchanged"] and coherent["protected_exact"]
for p, h in coherent["bindings"].items():
    assert sha(Path(p)) == h
assert all(v > 0 for v in read(root / "prefix-audit.json")["hidden_parameter_changes"].values())
summary = {}
for method in METHODS:
    norms = [r["metrics"][method]["gradient_norm"] for r in rows]
    summary[method] = dict(
        clipped=sum(v > 1 for v in norms),
        maximum_norm=max(norms),
        mean_norm=float(np.mean(norms)),
        last600_update_seconds=float(np.mean([r["seconds"][method] for r in rows[-600:]])),
    )
write(
    root / "diagnostics/integrity.json",
    dict(
        original_hh_batches_verified=1200,
        new_updates_per_arm=1200,
        inherited_updates=0,
        families=families,
        new_whole_axis_examples=whole,
        newly_allowed_partial_examples=promoted,
        parameters={m: 22508 for m in METHODS},
        reference_files_compared=counts,
        reference_arrays_compared=len(reference_errors),
        reference_arrays_nonexact=reference_nonexact,
        reference_maximum_absolute_difference=max(reference_errors),
        reference_tolerance=dict(atol=1e-5, rtol=1e-5),
        methods=summary,
    ),
)

files = [
    root / f
    for f in (
        "plan.json",
        "test-results.json",
        "teacher-reuse.json",
        "preflight.json",
        "prefix-audit.json",
        "snapshot-audit.json",
        "training/sampling.jsonl",
        "diagnostics/decision.json",
        "diagnostics/review-observations.json",
        "diagnostics/integrity.json",
    )
]
files += [root / f"driver-{phase}.json" for phase in phases]
files += [
    p
    for d in ("support-preview", "mask-probes", "coherent-probes")
    for p in (root / d).rglob("*")
    if p.is_file() and p.suffix in (".json", ".png")
]
files += [
    p
    for p in (root / "diagnostics").rglob("*")
    if p.is_file()
    and p.suffix in (".json", ".py", ".png")
    and p.name != "supplemental-binding.json"
]
files += [
    root / g / f"{n}.json"
    for g in ("evaluation", "stress", "scene512", "width-sweep")
    for n in ("metrics", "visual-review")
]
write(
    root / "completion.json",
    dict(
        stage="complete_late_opening",
        steps=1200,
        trained_arms=2,
        optimizer_reset_both=True,
        reference=(
            "HJ raw18000; matched control and opening fine-tuning; "
            "numeric tolerance declared before training"
        ),
        outputs=9516,
        additional_outputs=338,
        native_pages_reviewed=76,
        teacher_pages_reviewed=0,
        teacher_pages_reused=3,
        adoption=False,
        fresh_confirmation=False,
        evidence={str(p.relative_to(root)): sha(p) for p in set(files)},
    ),
)
print(dict(complete=True, integrity=read(root / "diagnostics/integrity.json")), flush=True)
