from contrast_composition_data import overlay_choice
from joint_geometry_study import SEED, STEPS, select
from optimized_broad_study import select as original


def test_fresh_shared_primary_sampling_and_independent_companions():
    assert SEED == 2026101440 and STEPS == 18000
    for step in range(60):
        chosen = select(step)
        for key, value in original(step).items():
            assert chosen[key] == value
        assert chosen["overlays"] == [
            overlay_choice(SEED, step, item, i) for item, i in enumerate(chosen["indices"])
        ]
