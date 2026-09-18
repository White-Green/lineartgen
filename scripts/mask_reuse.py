"""Freeze N1 control evidence and reuse its saved weights for the N2 comparison."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from summarize_training_methods import sha

from lineartgen.config import load_config
from lineartgen.training.comparison import comparison_configs


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def rows(path, through):
    result = [json.loads(line) for line in path.read_text().splitlines()][:through]
    assert [r["step"] for r in result] == list(range(1, through + 1))
    return result


def prepare(root, reference):
    assert not (root / "training").exists(), "Freeze before the main training starts"
    actual = comparison_configs(load_config(root / "config.toml"), study="mask_sampling")["control"]
    previous = load_config(reference / "training/control/config.json")
    assert actual == previous
    replay = root / "replay-control"
    assert load_config(replay / "control/config.json") == previous
    for name in ("dataset.json", "initialization.json"):
        assert read(replay / name) == read(reference / "training" / name)
    a, b = rows(replay / "timings.jsonl", 30), rows(reference / "training/timings.jsonl", 30)
    for left, right in zip(a, b, strict=True):
        for key in ("step", "scale_level", "recursion_levels"):
            assert left[key] == right[key]
        assert left["generated"]["control"] == right["generated"]["control"]
        np.testing.assert_array_equal(left["strengths"]["control"], right["strengths"]["control"])

    def logs(path):
        return {
            r["step"]: {k: v for k, v in r.items() if k != "update_seconds"}
            for line in path.read_text().splitlines()
            if (r := json.loads(line))["phase"] == "train" and r["step"] <= 30
        }

    x = logs(replay / "control/metrics.jsonl")
    y = logs(reference / "training/control/metrics.jsonl")
    assert set(x) == set(y) == {10, 20, 30}
    errors = []
    for step in x:
        assert set(x[step]) == set(y[step])
        for name, value in x[step].items():
            if isinstance(value, str):
                assert value == y[step][name]
            else:
                np.testing.assert_allclose(
                    value, y[step][name], atol=1e-8, rtol=1e-4 if name == "gradient_norm" else 1e-5
                )
                errors.append(abs(value - y[step][name]))
    source = read(reference / "training-source.json")
    changed = {
        "src/lineartgen/config.py",
        "src/lineartgen/diffusion.py",
        "src/lineartgen/evaluation.py",
        "src/lineartgen/training/comparison.py",
    }
    unchanged = {
        p: h
        for p, h in source["file_sha256"].items()
        if (p.startswith("src/") and p not in changed) or p in ("pyproject.toml", "uv.lock")
    }
    for p, h in unchanged.items():
        assert sha(Path(p)) == h, p
    assert sha(Path(source["archive"])) == source["archive_sha256"]
    files = [
        reference / "training-source.json",
        Path(source["archive"]),
        reference / "training/timings.jsonl",
        reference / "training/dataset.json",
        reference / "training/initialization.json",
        reference / "training/control/metrics.jsonl",
        reference / "training/control/config.json",
        reference / "training/control/dataset.json",
        replay / "dataset.json",
        replay / "initialization.json",
        replay / "timings.jsonl",
        replay / "control/metrics.jsonl",
        root / "uniform-equivalence.json",
    ]
    for step in (2000, 10000):
        for mode in ("raw", "ema"):
            files.extend((reference / f"snapshots/step-{step}/control/{mode}").iterdir())
            files.append(reference / f"evaluation/step-{step}-{mode}/metrics.json")
    files.extend((reference / "training/control/checkpoints/10000").rglob("*"))
    files.extend((reference / "training/initial/control").iterdir())
    write(
        root / "reference-reuse.json",
        dict(
            reference_root=str(reference),
            artifacts={str(p): sha(p) for p in files if p.is_file()},
            unchanged_numerical_source=unchanged,
            replay=dict(
                updates=30,
                logged_steps=[10, 20, 30],
                exact=x == y,
                max_error=max(errors),
                strength_scale_geometry_exact=True,
            ),
            limitation="One historical seed; no fresh control or paired speed claim. "
            "Defaults are bit-exact in input replay; update replay uses bounded FP32 tolerances.",
        ),
    )
    verify(root)


def verify(root):
    record = read(root / "reference-reuse.json")
    for path, digest in {**record["artifacts"], **record["unchanged_numerical_source"]}.items():
        assert sha(Path(path)) == digest, path
    reference = Path(record["reference_root"])
    assert comparison_configs(load_config(root / "config.toml"), study="mask_sampling")[
        "control"
    ] == load_config(reference / "training/control/config.json")
    return reference


def populate(root):
    """Copy immutable control evidence after the new run group has been initialized."""
    reference = verify(root)
    mappings = [
        ("training/initial/control", "training/initial/control"),
        ("training/control/checkpoints/10000", "training/control/checkpoints/10000"),
    ]
    mappings += [
        (f"snapshots/step-{step}/control/{mode}", f"snapshots/step-{step}/control/{mode}")
        for step in (2000, 10000)
        for mode in ("raw", "ema")
    ]
    for source, destination in mappings:
        for p in (reference / source).rglob("*"):
            if p.is_file():
                q = root / destination / p.relative_to(reference / source)
                q.parent.mkdir(parents=True, exist_ok=True)
                if not q.exists():
                    shutil.copyfile(p, q)
                assert sha(p) == sha(q)
    for name in ("config.json", "dataset.json"):
        p, q = reference / "training/control" / name, root / "training/control" / name
        if not q.exists():
            shutil.copyfile(p, q)
        assert sha(p) == sha(q)
    write(
        root / "training/control/REUSED_FROM.json",
        dict(
            reference_root=str(reference),
            reference_record_sha256=sha(root / "reference-reuse.json"),
            note="Copied evidence; this control was not trained in the N2 run.",
        ),
    )


def paired_timings(root, through):
    reference = verify(root)
    for name in ("dataset.json", "initialization.json"):
        assert read(root / "training" / name) == read(reference / "training" / name)
    actual = rows(root / "training/timings.jsonl", through)
    previous = rows(reference / "training/timings.jsonl", through)
    result = []
    for a, b in zip(actual, previous, strict=True):
        for key in ("step", "scale_level", "recursion_levels"):
            assert a[key] == b[key], (a["step"], key)
        assert set(a["updates"]) == {"boundary8"}
        assert a["generated"]["boundary8"] == b["generated"]["control"]
        np.testing.assert_array_equal(a["strengths"]["boundary8"], b["strengths"]["control"])
        result.append(
            dict(
                a,
                updates=dict(a["updates"], control=b["updates"]["control"]),
                generated=dict(a["generated"], control=b["generated"]["control"]),
            )
        )
    return result


def verify_reevaluation(root, report, step, mode):
    reference = verify(root)
    old = read(reference / f"evaluation/step-{step}-{mode}/metrics.json")
    actual = [
        {k: v for k, v in r.items() if not k.startswith("edit_boundary_")}
        for r in report["results"]
        if r["method"] == "control"
    ]
    expected = [r for r in old["results"] if r["method"] == "control"]
    assert actual == expected
    write(
        root / f"control-reevaluation-{step}-{mode}.json",
        dict(
            outputs=len(actual),
            original_metrics_exact=True,
            reference_sha256=sha(reference / f"evaluation/step-{step}-{mode}/metrics.json"),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root.resolve(), args.reference_root.resolve())
