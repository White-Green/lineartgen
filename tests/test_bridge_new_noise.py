import numpy as np
from bridge_new_noise import NEW_SEED, chunk_key
from velocity_bridge_toy import EVAL_SEED, measure, targets


def test_keys_are_stable_unique_and_separate_from_development_seed():
    assert NEW_SEED != EVAL_SEED
    values = []
    for depth in (1, 2, 3):
        for partial in (False, True):
            for strength in (0.1, 0.5):
                for chunk in range(16):
                    k = np.asarray(chunk_key(depth, partial, strength, chunk))
                    np.testing.assert_array_equal(
                        k, np.asarray(chunk_key(depth, partial, strength, chunk))
                    )
                    values.append(tuple(k))
    assert len(set(values)) == 192


def test_concatenated_metrics_preserve_draw_counts_and_averages():
    a, b, s = targets()
    p = np.stack([a, b, a, (a + b) / 2, b, b, a, a])
    source = np.broadcast_to(s, p.shape)
    t = np.ones_like(source) * 0.5
    rows = [measure(p[i : i + 2], a, b, source[i : i + 2], t[i : i + 2]) for i in range(0, 8, 2)]
    combined = measure(p, a, b, source, t)
    for key in ("mean_relative_endpoint_rms", "fraction_within_020", "teacher_averaged_mse"):
        np.testing.assert_allclose(combined[key], np.mean([r[key] for r in rows]), atol=1e-12)
    for key in ("nearest_A", "resolved_A", "resolved_B", "unresolved"):
        assert combined[key] == sum(r[key] for r in rows)
    assert combined["max_relative_endpoint_rms"] == max(
        r["max_relative_endpoint_rms"] for r in rows
    )


def test_quality_decision_is_invariant_to_replicating_the_same_draw_distribution():
    from bridge_new_noise import quality_checks

    row = dict(mean_relative_endpoint_rms=0.05, fraction_within_020=0.95, nearest_A=32)
    repeated = dict(row, nearest_A=512)
    assert all(quality_checks(row, 64).values())
    assert quality_checks(row, 64) == quality_checks(repeated, 1024)
