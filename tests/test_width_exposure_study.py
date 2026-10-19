from collections import Counter

import numpy as np
from attraction_augmentation import target_action
from attraction_graph import target
from group_field_model import INVERSE
from width_exposure_study import METHODS, STEPS, matched_batch, sampling


def test_sampling_depths_and_nested_exposure():
    rows = [sampling(i, 512) for i in range(STEPS)]
    assert Counter(r["depth"] for r in rows) == {i: 1000 for i in range(1, 7)}
    for method, p in zip(METHODS, (0.25, 0.5), strict=True):
        assert abs(sum(sum(r["replace"][method]) for r in rows) / (4 * STEPS) - p) < 0.02
    assert all(
        not x or y
        for r in rows
        for x, y in zip(r["replace"]["control"], r["replace"]["expanded"], strict=True)
    )
    assert 0.23 < sum(sum(r["partial"]) for r in rows) / (4 * STEPS) < 0.27
    assert all(rows[i] == sampling(i, 512) for i in (0, 1199, 2999, 5999))


def test_paired_items_and_teacher_follow_each_arm_replacement():
    examples = []
    for y in (61.2, 91.4, 151.1, 191.3):
        d = dict(
            source=np.ones((256, 256, 1), np.float32),
            graph_oracle=np.ones((320, 320, 1), np.float32),
        )
        d["source"][int(y) : int(y) + 2, 60:190] = 0.2
        for level in range(6):
            d[f"smalltarget{level}"], _ = target(
                np.array([[[92.2, y + 32], [221.3, y + 32]]]),
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
        if any(sampling(i, 2, 2)["shared_items"]) and not all(sampling(i, 2, 2)["shared_items"])
    )
    p, t, a = matched_batch(data, step)
    assert a["shared_items_exact"]
    assert a["input_sha256"]["control"] != a["input_sha256"]["expanded"]
    assert a["target_sha256"]["control"] != a["target_sha256"]["expanded"]
    for method in METHODS:
        assert len(p[method].noisy) == len(t[method]) == a["depth"]
        for level, labels in enumerate(t[method]):
            for k, (i, g) in enumerate(zip(a["indices"], a["groups"], strict=True)):
                d = (
                    data["extra"][a["extra_indices"][k]]
                    if a["replace"][method][k]
                    else data["legacy"][i]
                )
                np.testing.assert_array_equal(
                    target_action(np.asarray(labels[k]), INVERSE[g]), d[f"smalltarget{level}"]
                )
