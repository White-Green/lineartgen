import importlib.util
from pathlib import Path

import numpy as np

from lineartgen.config import DataConfig, TrainConfig
from lineartgen.data.pipeline import Source, split_experiment


def test_final_sources_are_disjoint_and_split_seed_is_independent():
    sources = [Source(str(i), ()) for i in range(12)]
    config = DataConfig(validation_count=3, final_test_count=2, split_seed=42)
    first = split_experiment(sources, config, 1)
    second = split_experiment(sources, config, 999)
    assert first == second
    sets = [set(x.name for x in group) for group in first]
    assert [len(x) for x in sets] == [7, 3, 2]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert set.union(*sets) == {str(i) for i in range(12)}
    train = TrainConfig(seed=42, model_seed=123)
    assert train.initialization_seed == 123 and train.seed == 42


def test_geometry_diagnostics_reject_cuts_bridges_and_widening():
    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_nested.py"
    spec = importlib.util.spec_from_file_location("evaluate_nested", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = np.ones((32, 32, 1), np.float32)
    target[4:28, 7:10] = -1
    target[4:28, 22:25] = -1
    noise = np.full_like(target, 0.5)
    noise[:, :16] = 0
    reference = module.geometry_metrics(target, target, noise)
    assert reference["ink_f1"] == 1 and reference["boundary_f1_1_5px"] == 1
    assert reference["component_count_error"] == reference["protected_mae"] == 0
    cut = target.copy()
    cut[14:18, 7:10] = 1
    bridge = target.copy()
    bridge[15:18, 10:22] = -1
    widened = target.copy()
    widened[4:28, 5:12] = -1
    for damaged in (cut, bridge, widened):
        result = module.geometry_metrics(damaged, target, noise)
        assert result["ink_f1"] < 1
        assert result["ink_area_error"] > 0
        assert result["protected_mae"] > 0
    assert module.geometry_metrics(cut, target, noise)["component_count_error"] == 1
    assert module.geometry_metrics(bridge, target, noise)["component_count_error"] == 1
