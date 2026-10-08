"""Use artificial scalar records only; never open the prospective confirmation inputs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from confirm_corruption_blend import active_rows, aggregate


def test_confirmation_gates_isolate_iterations_and_zero_strength():
    rows = []
    for steps in (1, 2, 5):
        for strength in (0.0, 0.1):
            for name in ("control", "candidate"):
                value = 0.8 if steps == 1 and strength > 0 else 20.0
                rows.append(
                    dict(
                        method=name,
                        domain="real",
                        family="strokes",
                        severity="strong",
                        partial=False,
                        kind="real",
                        strength=strength,
                        denoise_steps=steps,
                        halo_opacity_excess=1.0 if name == "control" else value,
                        active_ink_f1=0.9,
                    )
                )
    rules = dict(
        primary_baseline="control",
        halo_groups=["real/strokes/strong"],
        minimum_relative_halo_gain=0.1,
        safety_groups=["real/strokes/strong"],
        required_strata=["primary", "strength-0.1", "whole"],
        maximum_increases={},
        maximum_active_f1_drop=0.005,
    )
    assert len(active_rows(rows, 1)) == 2
    one = aggregate(rows, rules, {"candidate": "control"}, 1)
    five = aggregate(rows, rules, {"candidate": "control"}, 5)
    assert one["gates"]["candidate"]["numerical_gate_passed"]
    assert not five["gates"]["candidate"]["numerical_gate_passed"]
    assert all(r["denoise_steps"] == 1 and r["cases"] == 1 for r in one["summary"])


def test_each_seed_uses_its_own_paired_control():
    rows = [
        dict(
            method=name,
            domain="real",
            family="strokes",
            severity="strong",
            partial=False,
            kind="real",
            strength=0.1,
            denoise_steps=1,
            halo_opacity_excess=value,
        )
        for name, value in [("c07", 0.8), ("b07", 1.0), ("c06", 8.0), ("b06", 10.0)]
    ]
    rules = dict(
        halo_groups=["real/strokes/strong"],
        minimum_relative_halo_gain=0.1,
        safety_groups=[],
        required_strata=[],
        maximum_increases={},
        maximum_active_f1_drop=0.005,
    )
    result = aggregate(rows, rules, {"c07": "b07", "c06": "b06"}, 1)
    assert all(r["numerical_gate_passed"] for r in result["gates"].values())
    assert result["gates"]["c06"]["baseline"] == "b06"
