import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from assess_ink_structure_study import assess


def example():
    rules = dict(
        primary_baseline="control",
        halo_groups=["real/strokes/strong"],
        safety_groups=["real/strokes/strong"],
        required_strata=["primary"],
        minimum_relative_halo_gain=0.1,
        maximum_active_f1_drop=0.005,
        maximum_increases={"ink_deficit_fraction": 0.005},
    )
    rows = [
        dict(
            group="real/strokes/strong",
            method=m,
            halo_opacity_excess=h,
            ink_deficit_fraction=0.1,
            active_ink_f1=0.9,
        )
        for m, h in [("control", 0.2), ("candidate", 0.16)]
    ]
    return rows, rules


def test_halo_improvement_cannot_hide_line_loss():
    rows, rules = example()
    assert assess(rows, rules, "candidate")["numerical_gate_passed"]
    rows[1]["ink_deficit_fraction"] = 0.2
    assert not assess(rows, rules, "candidate")["numerical_gate_passed"]


def test_stable_lines_without_halo_improvement_do_not_pass():
    rows, rules = example()
    rows[1]["halo_opacity_excess"] = 0.195
    assert not assess(rows, rules, "candidate")["numerical_gate_passed"]


def test_missing_required_stratum_or_asymmetric_undefined_metric_is_an_error():
    rows, rules = example()
    rules["required_strata"].append("partial")
    with pytest.raises(ValueError, match="stratum absent"):
        assess(rows, rules, "candidate")
    rules["required_strata"] = ["primary"]
    rows[1]["ink_deficit_fraction"] = None
    with pytest.raises(ValueError, match="Mismatched"):
        assess(rows, rules, "candidate")
