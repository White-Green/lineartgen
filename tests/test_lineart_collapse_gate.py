import importlib.util
import json
from pathlib import Path

import pytest


def gate(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "lineart_collapse_test", scripts / "run_lineart_loss_study.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.collapse_failures


def reports(root, current):
    for step, rows in (
        (0, [dict(case=0, active_ink_f1=0.8), dict(case=1, active_ink_f1=None)]),
        (2000, current),
    ):
        folder = root / f"evaluation/step-{step}-ema"
        folder.mkdir(parents=True)
        (folder / "metrics.json").write_text(json.dumps(dict(results=rows)))


def test_identity_cases_do_not_hide_or_break_collapse_detection(tmp_path, monkeypatch):
    rows = [
        dict(method=method, case=case, family="clean", active_ink_f1=score if case == 0 else None)
        for method, score in (("control", 0.79), ("regions", 0.1))
        for case in (0, 1)
    ]
    reports(tmp_path, rows)
    assert gate(monkeypatch)(tmp_path, 2000) == [dict(method="regions", initial=0.8, current=0.1)]


def test_no_active_cases_is_not_a_pass(tmp_path, monkeypatch):
    reports(tmp_path, [dict(method="control", case=1, family="clean", active_ink_f1=None)])
    with pytest.raises(ValueError, match="No active"):
        gate(monkeypatch)(tmp_path, 2000)


def test_mismatched_active_cases_are_rejected(tmp_path, monkeypatch):
    reports(tmp_path, [dict(method="control", case=0, family="clean", active_ink_f1=None)])
    with pytest.raises(ValueError, match="Mismatched"):
        gate(monkeypatch)(tmp_path, 2000)
