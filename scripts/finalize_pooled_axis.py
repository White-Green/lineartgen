"""Verify actual training, matched-control predictions and all fixed native reviews."""

import argparse
import json
from pathlib import Path

import numpy as np
from campaign_protocol import read, sha, write
from pooled_axis_study import GP, GT, METHODS, SAVED, STEPS, check

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
root = parser.parse_args().root.resolve()
check(root)
rows = [json.loads(s) for s in (root / "training/sampling.jsonl").read_text().splitlines()]
gp = [json.loads(s) for s in (GP / "training/sampling.jsonl").read_text().splitlines()]
assert len(rows) == len(gp) == STEPS
families = {}
whole = 0
for k, (a, b) in enumerate(zip(rows, gp, strict=True)):
    assert a["step"] == k + 1
    assert a["audit"]["original_gp_input_sha256"] == b["input_sha256"]
    assert a["audit"]["original_gp_candidate_sha256"] == b["candidate_payload_sha256"]
    assert a["context_keep"] == b["context_keep"]
    family = a["audit"]["family"]
    families[family] = families.get(family, 0) + 1
    if family != "ambiguous":
        assert a["input_sha256"] == b["input_sha256"]
    if family == "ambiguous":
        whole += sum(a["audit"]["alternate_allowed"])
    if family == "long_bar_fill":
        assert not any(a["audit"]["alternate_allowed"])
    for method in METHODS:
        assert all(np.isfinite(v) for v in a["metrics"][method].values())
assert families == dict(legacy=3750, long_bar_fill=1500, ambiguous=750)
assert whole == 1966
for name in ("test-results", "preflight", "prefix-audit"):
    assert read(root / f"{name}.json")["passed"]
snap = read(root / "snapshot-audit.json")
assert snap["raw_ema_exact"] and snap["exports"] == 36
for f, h in snap["bindings"].items():
    assert sha(Path(f)) == h
for step in SAVED[1:]:
    assert read(root / f"training/restore-{step}.json")["full_state_exact"]
phases = [
    "preflight",
    *[f"train-{s}" for s in SAVED[3:]],
    "evaluation",
    "stress",
    "scene512",
    "width-sweep",
    "ambiguity",
    "geometry",
    "collision",
    "details",
    "orientation",
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
        with np.load(file) as a, np.load(GT / group / "predictions" / file.name) as b:
            for method, old_method in (("initial", "initial"),):
                np.testing.assert_array_equal(a[method], b[old_method])
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
    with np.load(file) as a, np.load(GT / "scene512" / file.name) as b:
        for method, old_method in (("initial", "initial"),):
            for encoding in ("float", "uint8"):
                np.testing.assert_array_equal(
                    a[method + "_" + encoding], b[old_method + "_" + encoding]
                )
for file in (root / "width-sweep").glob("wide-*.npz"):
    with np.load(file) as a, np.load(GT / "width-sweep" / file.name) as b:
        for key, old_key in (("source", "source"), ("initial", "initial")):
            np.testing.assert_array_equal(a[key], b[old_key])
for group, outputs, pages in [("ambiguity", 12, 3), ("geometry", 144, 9), ("collision", 18, 3)]:
    folder = root / "diagnostics" / group
    metrics = read(folder / "metrics.json")
    assert metrics["outputs"] == outputs and metrics["states_unchanged"]
    if group == "geometry":
        assert (
            metrics["primary_outputs"] == 72
            and metrics["repeated_anchors_exact"]
            and metrics["trained_anchors_match_FU_exact"]
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
            with np.load(file) as a, np.load(GT / "diagnostics" / group / file.name) as b:
                if "initial" in a.files:
                    np.testing.assert_array_equal(a["initial"], b["initial"])
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
        original_gp_batches_verified=6000,
        families=families,
        new_whole_axis_examples=whole,
        parameters={m: 21890 for m in METHODS},
        initial_reference_files_exact=counts,
        methods=summary,
    ),
)

files = [
    root / f
    for f in (
        "plan.json",
        "test-results.json",
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
        stage="complete_pooled_axis",
        steps=6000,
        trained_arms=2,
        reference="FW mixed6000; archived GT descriptive comparison",
        outputs=9516,
        additional_outputs=206,
        native_pages_reviewed=58,
        adoption=False,
        fresh_confirmation=False,
        evidence={str(p.relative_to(root)): sha(p) for p in set(files)},
    ),
)
print(dict(complete=True, integrity=read(root / "diagnostics/integrity.json")), flush=True)
