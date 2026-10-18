from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np
from attraction_augmentation import target_action
from attraction_graph import target
from attraction_objective import stage_loss as original_stage_loss
from fill_negative_objective import stage_loss
from fill_negative_study import METHODS, matched_batch, sampling
from group_field_model import INVERSE


def test_all_nonfill_losses_and_gradients_are_unchanged():
    h = jnp.linspace(-3, 2, 2 * 5 * 5 * 9).reshape(2, 5, 5, 9)
    t = jnp.zeros_like(h).at[:, 2, 2, [0, 4, 5, 6, 7, 8]].set(1)
    for level in (0, 1, 5):
        for part in ("vertex", "edge", "position", "width"):
            a = jax.value_and_grad(lambda x, level=level, part=part: stage_loss(x, t, level)[part])(
                h
            )
            b = jax.value_and_grad(
                lambda x, level=level, part=part: original_stage_loss(
                    x, t, level, sparse=True, base_radius=1
                )[part]
            )(h)
            for x, y in zip(a, b, strict=True):
                np.testing.assert_array_equal(x, y)


def test_sampling_depths_and_mixture():
    rows = [sampling(i, 512) for i in range(1200)]
    assert Counter(r["depth"] for r in rows) == {i: 200 for i in range(1, 7)}
    assert 0.22 < sum(sum(r["replace"]) for r in rows) / 4800 < 0.28
    assert 0.22 < sum(sum(r["partial"]) for r in rows) / 4800 < 0.28
    assert all(rows[i] == sampling(i, 512) for i in (0, 599, 1199))


def test_every_input_matches_and_teacher_follows_replaced_geometry():
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
    step = next(i for i in range(100) if any(sampling(i, 2, 2)["replace"]))
    p, t, a = matched_batch(data, step)
    assert a["all_items_exact"]
    assert a["input_sha256"][METHODS[0]] == a["input_sha256"][METHODS[1]]
    assert a["target_sha256"][METHODS[0]] == a["target_sha256"][METHODS[1]]
    for method in METHODS:
        assert len(p[method].noisy) == len(t[method]) == a["depth"]
        for level, labels in enumerate(t[method]):
            for k, (i, g) in enumerate(zip(a["indices"], a["groups"], strict=True)):
                d = data["extra"][a["extra_indices"][k]] if a["replace"][k] else data["legacy"][i]
                np.testing.assert_array_equal(
                    target_action(np.asarray(labels[k]), INVERSE[g]), d[f"smalltarget{level}"]
                )
