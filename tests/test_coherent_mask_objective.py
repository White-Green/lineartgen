import jax.numpy as jnp
import numpy as np
from coherent_mask_objective import axis_mask


def pair():
    fields = np.full((2, 18, 48, 48, 1), 10, np.float32)
    fields[0, :9, 8:40, 22:26] = -1
    fields[0, 9:, 22:26, 8:40] = -1
    fields[1, :9], fields[1, 9:] = fields[0, 9:].copy(), fields[0, :9].copy()
    return jnp.array([[1.0, 0.0], [0.0, -1.0]]), jnp.asarray(fields)


def test_whole_and_canonical_select_same_physical_axis_after_recipe_swap():
    normal, targets = pair()
    for t in (jnp.ones((2, 12, 12, 1)), jnp.zeros((2, 12, 12, 1))):
        m = np.asarray(axis_mask(normal, targets, t, "canonical"))
        assert not m[0, :9].any() and m[0, 9:].all()
        assert m[1, :9].all() and not m[1, 9:].any()
    np.testing.assert_array_equal(
        axis_mask(normal, targets, jnp.ones((2, 12, 12, 1)), "retained"),
        axis_mask(normal, targets, jnp.ones((2, 12, 12, 1)), "canonical"),
    )


def test_retained_prefers_active_vertical_tail_for_both_recipes():
    normal, targets = pair()
    t = np.zeros((2, 12, 12, 1), np.float32)
    t[:, 8:] = 0.5
    selected = np.asarray(axis_mask(normal, targets, jnp.asarray(t), "retained"))
    assert selected[0, :9].all() and not selected[0, 9:].any()
    assert not selected[1, :9].any() and selected[1, 9:].all()
    guard = np.asarray(axis_mask(normal, targets, jnp.asarray(t), "guard"))
    assert guard[:, :9].all() and not guard[:, 9:].any()


def test_retained_choice_does_not_depend_on_positive_strength_magnitude():
    normal, targets = pair()
    t = np.zeros((2, 12, 12, 1), np.float32)
    t[:, 8:] = 0.25
    np.testing.assert_array_equal(
        axis_mask(normal, targets, jnp.asarray(t), "retained"),
        axis_mask(normal, targets, jnp.asarray(3 * t), "retained"),
    )
