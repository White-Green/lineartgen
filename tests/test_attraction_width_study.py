from collections import Counter

import numpy as np
from attraction_augmentation import target_action
from attraction_graph import target
from attraction_width_study import matched_batch, sampling
from group_field_model import INVERSE


def test_matched_sampling_balances_depths_and_keeps_replacement_independent():
    rows = [sampling(i, 512) for i in range(3000)]
    assert Counter(r["depth"] for r in rows) == {i: 500 for i in range(1, 7)}
    assert all(rows[i] == sampling(i, 512) for i in (0, 1, 2999))
    assert set(i for r in rows for i in r["indices"]) == set(range(512))
    assert 0.22 < sum(sum(r["replace"]) for r in rows) / 12000 < 0.28
    assert 0.22 < sum(sum(r["partial"]) for r in rows) / 12000 < 0.28
    assert set(g for r in rows for g in r["groups"]) == set(range(8))


def test_unreplaced_items_match_and_replaced_targets_follow_selected_source():
    examples = []
    for y in (60.3, 90.4, 150.2, 190.1):
        d = dict(
            source=np.ones((256, 256, 1), np.float32),
            graph_oracle=np.ones((320, 320, 1), np.float32),
        )
        d["source"][int(y) : int(y) + 2, 70:190] = 0.2
        for level in range(6):
            d[f"smalltarget{level}"], _ = target(
                np.array([[[102.1, y + 32], [221.8, y + 32]]]),
                np.full((1, 2), 2.0),
                np.empty((0, 4)),
                320,
                level,
                base_radius=1,
            )
        examples.append(d)
    data = dict(legacy=examples[:2], extra=examples[2:])
    step = next(
        i
        for i in range(100)
        if any(sampling(i, 2, 2)["replace"]) and not all(sampling(i, 2, 2)["replace"])
    )
    p, ts, audit = matched_batch(data, step)
    assert audit["unchanged_items_exact"]
    assert audit["input_sha256"]["control"] != audit["input_sha256"]["expanded"]
    for method in ("control", "expanded"):
        assert len(p[method].noisy) == len(ts[method]) == audit["depth"]
        for level, a in enumerate(ts[method]):
            for k, (i, g) in enumerate(zip(audit["indices"], audit["groups"], strict=True)):
                d = (
                    data["extra"][audit["extra_indices"][k]]
                    if method == "expanded" and audit["replace"][k]
                    else data["legacy"][i]
                )
                np.testing.assert_array_equal(
                    target_action(np.asarray(a[k]), INVERSE[g]), d[f"smalltarget{level}"]
                )
