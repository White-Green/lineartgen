import numpy as np
from attraction_augmentation import matrix, points, spatial, target_action
from attraction_graph import decode, logits_from_target, target
from group_field_model import INVERSE, PRODUCT


def example():
    s = np.array([[[3.21, 5.37], [14.83, 13.19]], [[14.83, 13.19], [27.46, 7.82]]])
    w = np.array([[0.8, 1.2], [1.2, 2.0]])
    t, _ = target(s, w, np.empty((0, 4)), 32)
    return s, w, t


def test_d4_group_inverse_and_composition_preserve_edge_origins():
    _, _, t = example()
    for g in range(8):
        np.testing.assert_array_equal(target_action(target_action(t, g), INVERSE[g]), t)
        for h in range(8):
            np.testing.assert_array_equal(
                target_action(target_action(t, h), g), target_action(t, PRODUCT[g][h])
            )
            np.testing.assert_array_equal(matrix(g) @ matrix(h), matrix(PRODUCT[g][h]))


def test_label_action_matches_transformed_continuous_geometry():
    s, w, t = example()
    for g in range(8):
        expected, _ = target(points(s, 32, g), w, np.empty((0, 4)), 32)
        np.testing.assert_allclose(target_action(t, g), expected, atol=2e-6, rtol=2e-6)


def test_rendered_labels_rotate_and_reflect_without_boundary_wrapping():
    s, w, t = example()
    original, _ = decode(logits_from_target(t))
    for g in range(8):
        a, _ = decode(logits_from_target(target_action(t, g)))
        np.testing.assert_allclose(a, spatial(original, g), atol=1e-5, rtol=1e-5)
    # Boundary anchors exercise reversal origins at every canvas edge.
    t, _ = target(np.array([[[-1.0, 1.2], [8.0, 1.2]]]), np.ones((1, 2)), np.empty((0, 4)), 16)
    for g in range(8):
        np.testing.assert_array_equal(target_action(target_action(t, g), INVERSE[g]), t)


def test_fill_blank_and_vector_signs():
    t, _ = target(np.empty((0, 2, 2)), np.empty((0, 2)), np.array([[2, 4, 7, 11]]), 16)
    for g in range(8):
        np.testing.assert_array_equal(target_action(t, g)[..., 8], spatial(t[..., 8], g))
    np.testing.assert_array_equal(matrix(1) @ np.array([1, 0]), [0, -1])
    np.testing.assert_array_equal(matrix(4) @ np.array([1, 0]), [-1, 0])
