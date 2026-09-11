"""Repeated masks from one source must not count as independent observations."""

import importlib
from pathlib import Path

import pytest


def test_paired_bootstrap_resamples_sources_and_shapes(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    module = importlib.import_module("summarize_training_methods")
    cases, rows = [], []
    for suite in ("matched_real", "geometry"):
        for group in range(2):
            for condition in range(8):
                case = len(cases)
                cases.append(dict(source=f"source-{group}", kind="curve", variant=group))
                for method in module.METHODS:
                    score = 0.5 if method == "current" else 0.5 + (-0.2 if group == 0 else 0.2)
                    for subnet in range(3):
                        rows.append(
                            dict(
                                case=case,
                                suite=suite,
                                strength=(condition + 1) / 10,
                                active_target_ink_fraction=0.1,
                                method=method,
                                subnet=subnet,
                                denoise_steps=5,
                                ink_f1=score,
                                active_ink_f1=score,
                            )
                        )
    result = module.paired_deltas(dict(results=rows), dict(cases=cases))
    assert len(result) == 27
    for row in result:
        assert row["groups"] == 2 and row["cases"] == 16
        assert row["delta"] == pytest.approx(0)
        assert row["clustered_95_interval"] == pytest.approx([-0.2, 0.2])
