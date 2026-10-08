import jax
import jax.numpy as jnp
import numpy as np

from lineartgen.config import SketchAugmentationConfig
from lineartgen.sketch import augment_sketch


def test_mixed_examples_match_each_existing_family_and_share_enablement():
    count = 48
    clean = jnp.ones((count, 24, 24, 1)).at[:, 5:20, 10:13].set(-1)
    key = jax.random.PRNGKey(107)
    config = SketchAugmentationConfig(mode="mixed", probability=0.5)
    family = np.asarray(jax.random.randint(jax.random.fold_in(key, 4), (count,), 0, 3))
    enabled = np.asarray(jax.random.bernoulli(jax.random.fold_in(key, 0), 0.5, (count,)))
    actual = jax.jit(lambda x: augment_sketch(x, key, config))(clean)
    assert set(family[enabled]) == {0, 1, 2} and np.any(~enabled)
    for i, mode in enumerate(("tone", "blur", "strokes")):
        cfg = config.model_copy(update={"mode": mode})
        expected = jax.jit(lambda x, c=cfg: augment_sketch(x, key, c))(clean)
        np.testing.assert_allclose(actual[family == i], expected[family == i], rtol=0, atol=3e-7)
        # Choosing a family must not resample the clean/augmented switch.
        np.testing.assert_array_equal(np.asarray(expected)[~enabled], np.asarray(clean)[~enabled])
    np.testing.assert_array_equal(np.asarray(actual)[~enabled], np.asarray(clean)[~enabled])
    # The mixture chooses complete family outputs rather than averaging them.
    tone = augment_sketch(clean, key, config.model_copy(update={"mode": "tone"}))
    strokes = augment_sketch(clean, key, config.model_copy(update={"mode": "strokes"}))
    assert np.max(np.abs(np.asarray(tone)[enabled] - np.asarray(strokes)[enabled])) > 0.1
