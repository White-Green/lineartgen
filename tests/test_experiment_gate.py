import copy
import importlib.util
from pathlib import Path


def gate():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_training_methods.py"
    spec = importlib.util.spec_from_file_location("run_training_methods", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.pilot_gate


def report():
    methods = ["current", "residual", "ot_full", "independent_noise"]
    return {
        "checkpoint_step": 2000,
        "cases": 434,
        "results": [
            {"method": method, "protected_max_error": 0}
            for _ in range(434 * 3 * 3)
            for method in methods
        ],
        "summary": [
            dict(subset=subset, method=method, subnet=subnet, denoise_steps=5, ink_f1=0.7)
            for subset in ["matched_real", "geometry"]
            for method in methods
            for subnet in range(3)
        ],
    }


def test_gate_continues_learning_when_a_candidate_is_below_control():
    raw = report()
    raw["summary"][0]["ink_f1"] = 0.95
    assert gate()(raw, report())["passed"]


def test_gate_stops_for_missing_evaluation_collapse_or_broken_preservation():
    reference = report()
    incomplete = copy.deepcopy(reference)
    incomplete["results"].pop()
    collapsed = copy.deepcopy(reference)
    collapsed["summary"][-1]["ink_f1"] = 0.1
    altered = copy.deepcopy(reference)
    altered["results"][1]["protected_max_error"] = 0.01
    for invalid in (incomplete, collapsed, altered):
        assert not gate()(invalid, reference)["passed"]
