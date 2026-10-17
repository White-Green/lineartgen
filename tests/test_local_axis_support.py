import jax.numpy as jnp
import numpy as np
from consistent_axis_objective import candidate_mask
from local_axis_support import support_mask, support_safe


def case():
    t = np.full((1, 12, 12, 1), 0.5, np.float32)
    t[:, :3] = 0
    fields = np.full((1, 18, 48, 48, 1), 10, np.float32)
    fields[:, :, 28:36, 28:36] = -1
    return fields, t


def test_remote_mask_allows_same_axis_as_whole():
    fields, t = case()
    n = jnp.array([[1.0, 0.0]])
    actual = np.asarray(support_mask(n, jnp.asarray(fields), jnp.asarray(t)))
    whole = np.asarray(candidate_mask(n, jnp.ones_like(t), "consistent"))
    old = np.asarray(candidate_mask(n, jnp.asarray(t), "consistent"))
    np.testing.assert_array_equal(actual, whole)
    assert not np.array_equal(actual, old)


def test_one_shift_crossing_mask_blocks_alternative():
    fields, t = case()
    fields[:, 17, 11, 29] = 3.9  # Only one candidate's near-support is inactive.
    assert not bool(support_safe(jnp.asarray(fields), jnp.asarray(t))[0])


def test_missing_candidate_ink_blocks_blank_escape():
    fields, t = case()
    fields[:, 17] = 10
    assert not bool(support_safe(jnp.asarray(fields), jnp.asarray(t))[0])


def test_inactive_ink_and_all_blank_rejected():
    fields, t = case()
    t[:, 7:9, 7:9] = 0
    assert not bool(support_safe(jnp.asarray(fields), jnp.asarray(t))[0])
    assert not bool(support_safe(jnp.full_like(fields, 10), jnp.ones_like(t))[0])
