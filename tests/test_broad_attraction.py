from collections import Counter

import numpy as np
from attraction_augmentation import target_action
from attraction_graph import target
from broad_attraction_study import matched_batch, sampling
from group_field_model import INVERSE


def test_sampling_covers_shapes_depths_masks_and_d4():
    rows = [sampling(i, 512) for i in range(6000)]
    assert all(rows[i] == sampling(i, 512) for i in (0, 1, 5999))
    counts = Counter(i for r in rows for i in r["indices"])
    assert set(counts) == set(range(512)) and min(counts.values()) >= 20
    assert Counter(r["depth"] for r in rows) == {i: 1000 for i in range(1, 7)}
    assert set(g for r in rows for g in r["groups"]) == set(range(8))
    fraction = sum(sum(r["partial"]) for r in rows) / 24000
    assert 0.22 < fraction < 0.28


def test_both_target_radii_keep_item_alignment_after_preparation():
    data = []
    for y in (80.3, 170.4):
        d = dict(
            source=np.ones((256, 256, 1), np.float32),
            graph_oracle=np.ones((320, 320, 1), np.float32),
        )
        d["source"][int(y) : int(y) + 2, 70:190] = 0.2
        for prefix, radius in [("target", 4.0), ("smalltarget", 1.0)]:
            for level in range(6):
                d[f"{prefix}{level}"], _ = target(
                    np.array([[[102.1, y + 32], [221.8, y + 32]]]),
                    np.full((1, 2), 2.0),
                    np.empty((0, 4)),
                    320,
                    level,
                    base_radius=radius,
                )
        data.append(d)
    p, ts, audit = matched_batch(data, 5)
    assert len(p.noisy) == len(ts["control"]) == len(ts["sparse"]) == 6
    assert audit["target_sha256"]["control"] != audit["target_sha256"]["sparse"]
    for method, prefix in [("control", "target"), ("sparse", "smalltarget")]:
        for level, a in enumerate(ts[method]):
            assert np.isfinite(a).all()
            for k, (i, g) in enumerate(zip(audit["indices"], audit["groups"], strict=True)):
                np.testing.assert_array_equal(
                    target_action(np.asarray(a[k]), INVERSE[g]), data[i][f"{prefix}{level}"]
                )
