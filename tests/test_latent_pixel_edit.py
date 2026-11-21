import numpy as np
from lineartgen_runtime.images import PyramidPlan, prepare_input, to_bgra
from lineartgen_runtime.latent_context import max_pool
from lineartgen_runtime.latent_edit import PixelEdit
from lineartgen_runtime.latent_gap import facing_endpoints
from lineartgen_runtime.latent_painted_loops import loop_features
from lineartgen_runtime.latent_painted_paths import path_features, travel


def test_painted_rough_distance_distinguishes_adjacent_branches_and_disconnected_ink():
    ink = np.zeros((19, 43), bool)
    ink[4, 3:39] = ink[10, 3:39] = ink[15, 3:39] = True
    ink[4:11, 38] = True
    seed = np.zeros_like(ink)
    seed[4, 9] = True
    distance, chord = travel(ink, seed)
    assert distance[4, 27] == chord[4, 27] == 18
    assert distance[10, 10] > 5 * chord[10, 10]
    assert distance[15, 10] == -1
    rough = np.repeat(np.repeat(ink, 2, axis=0), 2, axis=1).astype(np.float32)
    mark = np.repeat(np.repeat(seed, 2, axis=0), 2, axis=1).astype(np.float32)
    values = path_features(rough[:-1, :-1], mark[:-1, :-1])
    assert values.shape == (*rough[:-1, :-1].shape, 6)
    assert not values[30].any()
    assert values[20, 20, 1] > 0.75 and values[8, 54, 1] == 0


def test_missing_stroke_can_have_facing_ends_while_parallel_lines_have_none():
    ink = np.zeros((1, 64, 72, 1), np.float32)
    ink[:, 25:27, 3:69] = 0.6
    ink[:, 37:39, 3:69] = 0.5

    def peak3(x):
        return max_pool(x[0], 3)[None]

    evidence = facing_endpoints(ink, np, peak3)
    assert evidence[0, 32, 36, 0] < 0.01
    ink[:, 37:39] = 0
    ink[:, 25:27, 31:41] = 0
    assert facing_endpoints(ink, np, peak3)[0, 26, 36, 0] > 0.9
    rotated = np.rot90(ink, axes=(1, 2))
    np.testing.assert_allclose(
        np.rot90(facing_endpoints(rotated, np, peak3), -1, axes=(1, 2))[0, 26, 36, 0],
        facing_endpoints(ink, np, peak3)[0, 26, 36, 0],
        atol=0.01,
    )


def test_masked_latent_proposal_preserves_existing_ink_white_and_partial_coverage():
    h, w = 9, 13
    rng = np.random.default_rng(65)
    rough = rng.integers(0, 256, size=(h, w, 4), dtype=np.uint8)
    layer = np.zeros_like(rough)
    layer[:, 1:4] = (0, 0, 0, 255)
    layer[:, 4:6] = 255
    layer[:, 6:8] = (160, 80, 90, 255)
    layer[:, 8:11] = (0, 0, 0, 128)
    layer[:, 11:] = (255, 255, 255, 128)
    plan = PyramidPlan((h, w), (h, w), ((h, w),))
    clean, power = prepare_input(rough.tobytes(), layer.tobytes(), plan, 0.5, image_scale=1.0)
    reference = to_bgra(clean, plan, alpha_threshold=0)
    under = np.full((h, w), 0.4, np.float32)
    edit = PixelEdit(rough.tobytes(), layer.tobytes(), h, w, 0.5)
    assert edit.render() == reference
    mask = power[0, ..., 0] == 0
    output = np.frombuffer(edit.render(under), np.uint8).reshape(h, w, 4)
    expected = np.frombuffer(reference, np.uint8).reshape(h, w, 4)
    np.testing.assert_array_equal(output[mask], expected[mask])
    np.testing.assert_array_equal(output[:, 0, 3], 102)
    np.testing.assert_array_equal(output[:, 8:11, 3], 179)
    np.testing.assert_array_equal(output[:, 11:, 3], 51)
    stopped = PixelEdit(rough.tobytes(), layer.tobytes(), h, w, 0)
    assert stopped.render(under) == reference


def test_stroke_along_one_side_of_a_closed_rough_identifies_the_opposite_side():
    rough = np.zeros((71, 121), np.float32)
    rough[28:30, 12:109] = rough[40:42, 12:109] = 0.5
    rough[28:42, 12:14] = rough[28:42, 107:109] = 0.5
    painted = np.zeros_like(rough)
    painted[28:30, 45:53] = 1
    sides = loop_features(rough, painted)[..., 0]
    assert sides[28, 75] > 0.95 and sides[41, 76] < -0.95
    assert not np.any(loop_features(rough, np.zeros_like(painted)))
    rough[28:42, 12:15] = 0
    assert not np.any(loop_features(rough, painted))
