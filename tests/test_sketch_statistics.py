import importlib
from pathlib import Path

import pytest


def test_sketch_intervals_resample_sources_and_weight_case_counts(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    module = importlib.import_module("summarize_sketch_experiment")
    result = module.clustered_delta({"source-a": [-0.2] * 8, "source-b": [0.2] * 8})
    assert result["groups"] == 2 and result["cases"] == 16
    assert result["delta"] == pytest.approx(0)
    assert result["clustered_95_interval"] == pytest.approx([-0.2, 0.2])
    unequal = module.clustered_delta({"source-a": [-0.2] * 12, "source-b": [0.2] * 4})
    assert unequal["delta"] == pytest.approx(-0.1)
    assert module.clustered_delta({}) is None


def test_pairing_separates_common_initialization_and_control(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    module = importlib.import_module("summarize_sketch_experiment")
    rows, initial, cases = [], [], []
    for i in range(8):
        cases.append(dict(kind="real", source=f"source-{i // 4}"))
        for method, score in (("initial", 0.5), ("control", 0.6), ("tone", 0.8)):
            row = dict(
                case=i,
                suite="sketch_real",
                family="tone",
                severity="mild",
                strength=0.1,
                method=method,
                subnet=0,
                denoise_steps=1,
                active_ink_f1=score,
                active_mae=1 - score,
            )
            (initial if method == "initial" else rows).append(row)
    result = module.paired_deltas(dict(results=rows), dict(results=initial), dict(cases=cases))
    assert len(result) == 6
    for row in result:
        assert row["groups"] == 2 and row["cases"] == 8
        expected = 0.2 if row["baseline"] == "control" else 0.3 if row["method"] == "tone" else 0.1
        if row["metric"] == "active_mae":
            expected *= -1
        assert row["delta"] == pytest.approx(expected)
