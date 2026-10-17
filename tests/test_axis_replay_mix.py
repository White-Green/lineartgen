import numpy as np
from axis_replay_mix import background_indices, select


def test_interleaving_and_matched_depths():
    ids = background_indices()
    assert len(ids) == len(set(ids)) == 1200
    assert all((i // 6) % 8 != 5 for i in ids)
    for k in range(1200):
        assert select(2 * k, ids) == ("focus", k, k % 6 + 1)
        assert select(2 * k + 1, ids) == ("background", ids[k], k % 6 + 1)
    for family in ("focus", "background"):
        assert [
            sum(select(k, ids)[0] == family and select(k, ids)[2] == d for k in range(2400))
            for d in range(1, 7)
        ] == [200] * 6


def test_focus_examples_exposure_is_gx():
    visits = np.zeros((12, 6), int)
    for k in range(1200):
        visits[(k // 6 % 3) * 4 : (k // 6 % 3) * 4 + 4, k % 6] += 1
    assert visits.sum() == 4800
    assert sorted(visits.sum(1).tolist()) == [396] * 4 + [402] * 8
    assert visits.max() - visits.min() <= 1
