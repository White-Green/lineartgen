import copy
import importlib.util
import json
from pathlib import Path

import pytest


def module(name, monkeypatch):
    folder = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location(name, folder / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_campaign_grid_rejects_missing_duplicate_nonfinite_and_protected(tmp_path, monkeypatch):
    p = module("campaign_protocol", monkeypatch)
    cases = tmp_path / "cases"
    cases.mkdir()
    p.write(cases / "cases.json", {"cases": [{}, {}, {}]})
    arm = tmp_path / "arm"
    p.write(arm / "config.json", {"model": {"nested_channels": []}})
    report = dict(
        checkpoint_step=10000,
        ema=False,
        cases=3,
        manifest_sha256=p.sha(cases / "cases.json"),
        noise_generator="numpy",
        recursion="runtime",
        results=[
            dict(
                case=i,
                method="model",
                subnet=0,
                denoise_steps=1,
                recursion_levels=3,
                protected_max_error=0.0,
                active_mae=0.1,
            )
            for i in range(3)
        ],
    )
    destination = arm / "evaluation/step-10000-raw/metrics.json"
    p.write(destination, report)
    assert p.report(arm, 10000, "raw", cases) == report
    for issue in ("missing", "duplicate", "nonfinite", "protected", "manifest"):
        bad = copy.deepcopy(report)
        if issue == "missing":
            bad["results"].pop()
        elif issue == "duplicate":
            bad["results"][-1] = bad["results"][0]
        elif issue == "nonfinite":
            bad["results"][0]["active_mae"] = float("nan")
        elif issue == "protected":
            bad["results"][0]["protected_max_error"] = 1e-8
        else:
            bad["manifest_sha256"] = "different"
        destination.write_text(json.dumps(bad))
        with pytest.raises(AssertionError):
            p.report(arm, 10000, "raw", cases)


def test_resume_preserves_uncommitted_log_bytes_and_replays_from_checkpoint(tmp_path, monkeypatch):
    runner = module("run_completion_campaign", monkeypatch)
    campaign = object.__new__(runner.Campaign)
    events = []
    campaign.status = lambda *a, **k: events.append((a, k))
    arm = tmp_path / "arm"
    path = arm / "training/timings.jsonl"
    path.parent.mkdir(parents=True)
    original = '{"step":1}\n{"step":2}\n{"step":3}\n{"step":'
    path.write_text(original)
    campaign.recover_logs(arm, 2)
    assert path.read_text() == '{"step":1}\n{"step":2}\n'
    assert next(path.parent.glob("*.uncommitted-*")).read_text() == original
    campaign.recover_logs(arm, 2)
    assert len(events) == 1
