"""Teacher-only round-trip check for a coverage-preserving distance extension."""

import argparse
import shutil
from pathlib import Path

import numpy as np
from campaign_protocol import read, sha, write
from diagnose_learnability import freeze, verify
from paper_band_fit import PROBE, TRAIN, fit_checks
from PIL import Image, ImageDraw

from lineartgen.coverage_metrics import coverage_metrics
from lineartgen.evaluation import probe_metrics, protected_metrics
from lineartgen.lineart_metrics import halo_metrics
from lineartgen.training.coverage_distance import (
    binary_distance,
    coverage_distance,
    render_distance,
)

DOC = Path("docs/coverage-distance-oracle-plan-20261011.md")
EXTRAS = ("paper", "black", "faint", "half", "gap1", "alpha256")
METHODS = ("binary", "coverage")


def extra_targets():
    paper = np.ones((512, 512, 1), np.float32)
    faint, half, gap, alpha = (paper.copy() for _ in range(4))
    faint[100:412, 256:257] = 0.6
    half[100:412, 256:257] = 0
    gap[255:256, 100:256] = -1
    gap[255:256, 257:412] = -1
    tile = (np.arange(256, dtype=np.float32) / 255 * 2 - 1).reshape(16, 16)
    alpha[192:320, 192:320, 0] = np.repeat(np.repeat(tile, 8, axis=0), 8, axis=1)
    return dict(zip(EXTRAS, (paper, -paper, faint, half, gap, alpha), strict=True))


def prepare(root):
    assert not root.exists()
    (root / "initial").mkdir(parents=True)
    (root / "configs").mkdir()
    parent = Path("../jax-paper-band-fit-20261010/runs/paper-band-fit-20261010").resolve()
    verify(parent)
    shutil.copyfile(parent / "inputs.npz", root / "inputs.npz")
    np.savez_compressed(root / "extra.npz", **extra_targets())
    write(root / "train.json", dict(updates=0, teacher_only=True))
    write(root / "probe.json", dict(fixed_ids=list(TRAIN + PROBE), extras=list(EXTRAS)))
    write(
        root / "plan.json",
        dict(
            initial=str(root / "initial"),
            initial_files={},
            data={
                n: dict(path=str(root / n), sha256=sha(root / n))
                for n in ("inputs.npz", "extra.npz")
            },
            parent=str(parent),
            parent_source_sha256=sha(parent / "source.json"),
            crops=read(parent / "plan.json")["crops"],
            radius=8,
            updates=0,
            plan_document_sha256=sha(DOC),
            adoption=False,
        ),
    )


def check(root):
    verify(root)
    plan = read(root / "plan.json")
    assert sha(DOC) == plan["plan_document_sha256"]
    verify(Path(plan["parent"]))
    assert sha(Path(plan["parent"]) / "source.json") == plan["parent_source_sha256"]
    return plan


def inputs(root):
    with np.load(root / "inputs.npz") as data:
        cases = [
            (
                i,
                data[f"{i}-image"],
                data[f"{i}-source"],
                (data[f"{i}-probes"], data[f"{i}-connections"]),
            )
            for i in TRAIN + PROBE
        ]
    with np.load(root / "extra.npz") as data:
        cases += [(i, data[i], 1 - 0.3 * (1 - data[i]), None) for i in EXTRAS]
    return cases


def evaluate(root):
    plan = check(root)
    assert read(root / "test-results.json")["passed"]
    out = root / "evaluation"
    (out / "predictions").mkdir(parents=True)
    rows, checks, bindings = [], [], {}
    displays, fields = {}, {}
    for case, clean, source, probes in inputs(root):
        strength = np.full_like(clean, 0.5)
        partial = strength.copy()
        partial[:, :256] = 0
        masked_target = np.where(partial > 0, clean, source)
        displays[(case, "target")] = clean
        for method, fn in (("binary", binary_distance), ("coverage", coverage_distance)):
            field = fn(clean, plan["radius"])
            prediction = np.asarray(render_distance(field))
            masked = np.asarray(render_distance(field, source, partial))
            assert np.isfinite(field).all() and np.max(np.abs(field)) <= plan["radius"]
            assert np.isfinite(prediction).all() and np.max(np.abs(prediction)) <= 1
            np.testing.assert_array_equal(masked[:, :256], source[:, :256])
            metrics = protected_metrics(prediction, clean, strength)
            metrics |= halo_metrics(prediction, clean, strength)
            metrics |= coverage_metrics(prediction, clean, strength)
            if probes is not None:
                metrics |= probe_metrics(prediction, clean, *probes)
            row = dict(
                method=method,
                case=case,
                split="train" if case in TRAIN else "probe",
                maximum_error=float(np.max(np.abs(prediction - clean))),
                masked_error=float(np.max(np.abs(masked - masked_target))),
                **metrics,
            )
            rows.append(row)
            if method == "coverage":
                checks.append(
                    dict(case=case, metric="max_error", passed=row["maximum_error"] <= 1e-6)
                )
                checks.append(
                    dict(case=case, metric="masked_error", passed=row["masked_error"] <= 1e-6)
                )
                if probes is not None:
                    checks += fit_checks([row])
            path = out / f"predictions/{case}-{method}.npz"
            np.savez_compressed(path, field=field, prediction=prediction, masked=masked)
            bindings[str(path)] = sha(path)
            displays[(case, method)] = prediction
            fields[(case, method)] = field
    assert len(rows) == len(bindings) == 28 and len(checks) == 50
    write(
        out / "metrics.json",
        dict(
            rows=rows,
            checks=checks,
            bindings=bindings,
            passed=all(c["passed"] for c in checks),
            finite=True,
            protected_exact=True,
            adoption=False,
        ),
    )
    records = []
    for label, ids in (("geometry", TRAIN + PROBE), ("extras", EXTRAS)):
        canvas = Image.new("L", (640, 184 * len(ids)), 255)
        for row, case in enumerate(ids):
            values = [displays[(case, m)] for m in ("target", "binary", "coverage")]
            values.append(1 - 2 * np.clip(4 * np.abs(values[1] - values[0]), 0, 1))
            for col, (name, value) in enumerate(
                zip(("target", "binary", "coverage", "binary_error_x4"), values, strict=True)
            ):
                tile = Image.fromarray(np.rint((value[..., 0] + 1) * 127.5).astype(np.uint8))
                crop = plan["crops"][str(case)] if label == "geometry" else [176, 176, 336, 336]
                canvas.paste(tile.crop(crop), (col * 160, row * 184 + 24))
                ImageDraw.Draw(canvas).text(
                    (col * 160 + 2, row * 184 + 4), f"{case} {name}", fill=0
                )
        path = out / f"{label}-native.png"
        canvas.save(path)
        records.append(dict(file=path.name, sha256=sha(path)))
    canvas = Image.new("L", (480, 184 * 8), 255)
    for row, case in enumerate(TRAIN + PROBE):
        values = [displays[(case, "target")], *[fields[(case, m)] / 8 for m in METHODS]]
        for col, (label, value) in enumerate(
            zip(("target", "binary_distance", "coverage_distance"), values, strict=True)
        ):
            tile = Image.fromarray(np.rint((value[..., 0] + 1) * 127.5).astype(np.uint8))
            canvas.paste(tile.crop(plan["crops"][str(case)]), (col * 160, row * 184 + 24))
            ImageDraw.Draw(canvas).text((col * 160 + 2, row * 184 + 4), f"{case} {label}", fill=0)
    path = out / "fields-native.png"
    canvas.save(path)
    records.append(dict(file=path.name, sha256=sha(path)))
    write(out / "render-manifest.json", dict(records=records))
    write(root / "workflow-status.json", dict(stage="oracle_complete_pending_review", updates=0))


def finalize(root):
    check(root)
    tests = read(root / "test-results.json")
    assert tests["passed"]
    for p, digest in tests["logs"].items():
        assert sha(root / p) == digest
    m = read(root / "evaluation/metrics.json")
    assert len(m["rows"]) == 28 and len(m["checks"]) == 50 and len(m["bindings"]) == 28
    assert m["finite"] and m["protected_exact"]
    for p, digest in m["bindings"].items():
        assert sha(Path(p)) == digest
        with np.load(p) as a:
            assert all(v.shape == (512, 512, 1) and np.isfinite(v).all() for v in a.values())
            np.testing.assert_array_equal(a["prediction"], np.clip(2 * a["field"], -1, 1))
    review = read(root / "evaluation/visual-review.json")
    render = read(root / "evaluation/render-manifest.json")
    assert review["reviewed"] and len(review["files"]) == len(render["records"]) == 3
    for r in render["records"]:
        assert sha(root / "evaluation" / r["file"]) == r["sha256"] == review["files"][r["file"]]
    result = dict(
        stage="complete_coverage_distance_oracle",
        updates=0,
        adoption=False,
        numerical_commit=read(root / "source.json")["commit"],
        round_trip_passed=m["passed"],
        checks=50,
        final_images_reviewed=3,
        learned_quality_established=False,
        exact_continuous_sdf=False,
        evidence={
            p: sha(root / p)
            for p in (
                "source.json",
                "plan.json",
                "test-results.json",
                "evaluation/metrics.json",
                "evaluation/visual-review.json",
            )
        },
    )
    write(root / "final-audit.json", result)
    write(root / "workflow-status.json", result)
    print(result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "freeze", "evaluate", "finalize"))
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    {"prepare": prepare, "freeze": freeze, "evaluate": evaluate, "finalize": finalize}[args.action](
        args.root.resolve()
    )
