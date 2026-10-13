import jax
import numpy as np
from expanded_family_data import make_case
from expansion_study import selection, training_batch


def test_new_variants_are_distinct_but_keep_the_same_problem():
    for slot in (0, 6, 12, 18, 24, 30, 36, 42, 48, 58):
        a, ar = make_case("expanded", slot, 1)
        b, br = make_case("new_probe", slot, 0)
        assert ar["variants_sha256"] != br["variants_sha256"]
        assert ar["kind"] == br["kind"]
        assert a["variants"].shape == b["variants"].shape == (27, 256, 256, 1)
        assert np.isfinite(a["source"]).all() and a["source"].min() >= 0.2 - 1e-6


def test_matching_slots_masks_noise_and_corpus_indices():
    values = np.arange(512, dtype=np.float32) / 512
    data = dict(
        variants=np.broadcast_to(values[:, None, None, None, None], (512, 27, 256, 256, 1)),
        source=np.broadcast_to(values[:, None, None, None], (512, 256, 256, 1)),
    )
    for step in range(6):
        s = selection(step)
        fixed, a = training_batch(data, step, "fixed")
        expanded, b = training_batch(data, step, "expanded")
        np.testing.assert_array_equal(
            np.asarray(a)[:, 0, 0, 0, 0], values[np.array(s["indices"]) * 8]
        )
        np.testing.assert_array_equal(
            np.asarray(b)[:, 0, 0, 0, 0], values[np.array(s["indices"]) * 8 + s["variants"]]
        )
        jax.tree.map(np.testing.assert_array_equal, fixed.noise_levels, expanded.noise_levels)
        assert len(fixed.noisy) == len(expanded.noisy) == step + 1
        for fa, fg, ea, eg in zip(
            fixed.noisy, fixed.guides, expanded.noisy, expanded.guides, strict=True
        ):
            np.testing.assert_allclose(
                np.asarray(fa - fg), np.asarray(ea - eg), atol=1.2e-7, rtol=0
            )
