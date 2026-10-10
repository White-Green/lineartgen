import importlib.util
from pathlib import Path

import evaluate_procedural_structure as evaluator
import pytest
from campaign_protocol import read, write


@pytest.mark.parametrize("control_name", ["pixel", "band4000"])
def test_explicit_control_preserves_all_existing_local_checks(tmp_path, control_name):
    old_root = Path(
        "/home/whitegreen/projects/lineartgen/.worktrees/jax-paper-band-long-run-20261010"
    )
    run = old_root / "runs/paper-band-long-run-20261010"
    spec = importlib.util.spec_from_file_location(
        "frozen_local_evaluation", old_root / "scripts/evaluate_procedural_structure.py"
    )
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)

    def rename(value, before, after):
        if isinstance(value, dict):
            return {after if k == before else k: rename(v, before, after) for k, v in value.items()}
        if isinstance(value, list):
            return [rename(v, before, after) for v in value]
        return after if value == before else value

    for prefix in ("broad", "evaluation"):
        rows = read(run / f"{prefix}/step-4000/metrics.json")["rows"]
        write(
            tmp_path / f"{prefix}/step-4000/metrics.json",
            dict(rows=rename(rows, "band4000", "pixel")),
        )
    write(tmp_path / "plan.json", dict(comparison_methods=["pixel", "band"]))
    old.summarize(tmp_path, 4000)
    expected = read(tmp_path / "broad/step-4000/summary.json")
    for prefix in ("broad", "evaluation"):
        path = tmp_path / f"{prefix}/step-4000/metrics.json"
        write(path, rename(read(path), "pixel", control_name))
    plan = dict(comparison_methods=[control_name, "band"])
    if control_name != "pixel":
        plan["local_control_method"] = control_name
    write(tmp_path / "plan.json", plan)
    evaluator.summarize(tmp_path, 4000)
    actual = read(tmp_path / "broad/step-4000/summary.json")
    actual.pop("metrics_sha256")
    expected.pop("metrics_sha256")
    assert actual == rename(expected, "pixel", control_name)
