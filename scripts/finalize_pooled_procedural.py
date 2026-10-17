"""Verify actual training, matched-control predictions and all fixed native reviews."""

import argparse
import json
from pathlib import Path

import numpy as np
from campaign_protocol import read, sha, write
from pooled_procedural_study import GM, SAVED, STEPS, check

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
root = parser.parse_args().root.resolve()
check(root)
rows = [json.loads(s) for s in (root / "training/sampling.jsonl").read_text().splitlines()]
gm = [json.loads(s) for s in (GM / "training/sampling.jsonl").read_text().splitlines()]
assert len(rows) == len(gm) == STEPS
for k, (a, b) in enumerate(zip(rows, gm, strict=True)):
    assert a["step"] == k + 1
    assert a["input_sha256"] == b["input_sha256"]["procedural"]
    assert a["candidate_payload_sha256"] == b["candidate_payload_sha256"]["procedural"]
    assert a["context_keep"] == b["context_keep"]
    assert all(np.isfinite(v) for v in a["metrics"]["pooled"].values())
for name in ("test-results", "preflight", "prefix-audit"):
    assert read(root / f"{name}.json")["passed"]
snap = read(root / "snapshot-audit.json")
assert snap["raw_ema_exact"] and snap["exports"] == 18
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
        with np.load(file) as a, np.load(GM / group / "predictions" / file.name) as b:
            for method in ("initial", "procedural"):
                np.testing.assert_array_equal(a[method], b[method])
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
    with np.load(file) as a, np.load(GM / "scene512" / file.name) as b:
        for method in ("initial", "procedural"):
            for encoding in ("float", "uint8"):
                np.testing.assert_array_equal(
                    a[method + "_" + encoding], b[method + "_" + encoding]
                )
for file in (root / "width-sweep").glob("wide-*.npz"):
    with np.load(file) as a, np.load(GM / "width-sweep" / file.name) as b:
        for key in ("source", "initial", "procedural"):
            np.testing.assert_array_equal(a[key], b[key])
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
    if group in ("ambiguity", "geometry"):
        for file in folder.glob("*.npz"):
            with np.load(file) as a, np.load(GM / "diagnostics" / group / file.name) as b:
                np.testing.assert_array_equal(a["procedural"], b["procedural"])
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
norms = [r["metrics"]["pooled"]["gradient_norm"] for r in rows]
write(
    root / "diagnostics/integrity.json",
    dict(
        all6000_gm_payloads_exact=True,
        parameters=dict(procedural=21718, pooled=21778),
        reference_prediction_files_exact=counts,
        clipped=sum(v > 1 for v in norms),
        maximum_norm=max(norms),
        mean_norm=float(np.mean(norms)),
        last600_update_seconds=float(np.mean([r["seconds"]["pooled"] for r in rows[-600:]])),
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
        stage="complete_pooled_procedural",
        steps=6000,
        trained_arms=1,
        reused_control="GM procedural6000",
        outputs=9516,
        additional_outputs=174,
        native_pages_reviewed=54,
        adoption=False,
        fresh_confirmation=False,
        evidence={str(p.relative_to(root)): sha(p) for p in set(files)},
    ),
)
print(dict(complete=True, integrity=read(root / "diagnostics/integrity.json")), flush=True)
