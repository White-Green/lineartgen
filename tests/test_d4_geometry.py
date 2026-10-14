import jax
import jax.numpy as jnp
import numpy as np
import subpixel_field_study as study
from d4_geometry_data import transform_batch, transform_directions, transform_numpy_batch
from regional_field_objective import field_parts
from rough_family_data import BASE
from subpixel_field_data import candidate_fields
from test_focused_geometry import data
from test_subpixel_field import fixture


def test_dynamic_batch_transforms_match_cpu_and_invert():
    x = np.arange(8 * 32 * 32 * 3, dtype=np.float32).reshape(8, 32, 32, 3)
    codes = np.arange(8, dtype=np.int32)
    y = transform_batch(jnp.asarray(x), codes)
    np.testing.assert_array_equal(y, transform_numpy_batch(x, codes))
    np.testing.assert_array_equal(transform_batch(y, codes, inverse=True), x)
    np.testing.assert_array_equal(transform_numpy_batch(np.asarray(y), codes, inverse=True), x)
    directions = np.tile(np.array([[0.6, 0.8]], np.float32), (8, 1))
    np.testing.assert_array_equal(
        transform_directions(transform_directions(directions, codes), codes, inverse=True),
        directions,
    )


def test_candidate_geometry_and_loss_are_equivariant_at_six_scales():
    _, base = fixture()
    directions = np.array([[0.6, 0.8]], np.float32)
    targets = candidate_fields(base, jnp.asarray(directions), 6)
    fields = tuple(t[:, 4] + 0.3 * jnp.sin(t[:, 4] * 2) for t in targets)
    strengths = tuple(jnp.full((1, f.shape[1] // 4, f.shape[2] // 4, 1), 0.5) for f in fields)
    reference = field_parts(fields, targets, strengths)
    for code in range(8):
        codes = np.array([code], np.int32)
        rotated = transform_batch(base[..., None], codes)[..., 0]
        moved = transform_directions(directions, codes)
        actual = candidate_fields(rotated, jnp.asarray(moved), 6)
        for a, b in zip(targets, actual, strict=True):
            shape = a.shape
            expected = transform_batch(a.reshape((-1,) + shape[2:]), np.repeat(codes, 9))
            np.testing.assert_allclose(b, expected.reshape(shape), atol=2e-5, rtol=0)
        value = field_parts(
            tuple(transform_batch(f, codes) for f in fields),
            actual,
            tuple(transform_batch(t, codes) for t in strengths),
        )
        for key in reference:
            np.testing.assert_allclose(value[key], reference[key], atol=2e-6, rtol=2e-5)


def test_training_batches_match_after_inverse_including_noise_and_masks(monkeypatch):
    arrays = data()
    original = study.strength_map
    monkeypatch.setattr(study, "strength_map", lambda mode: original(mode)[::8, ::8])
    for depth in range(1, 7):
        monkeypatch.setattr(
            study,
            "select",
            lambda step, depth=depth: dict(
                indices=[0] * 4,
                modes=[0, 1, 2, 0],
                depth=depth,
                augmentation=[BASE, BASE | dict(opacity=0.2, sigma=2.4, amplitude=1.5)] * 2,
                transforms=[0, 1, 4, 7],
            ),
        )
        a, b = (study.batch(arrays, depth, m) for m in study.METHODS)
        c = study.canonical_batch(*b, depth, "d4")
        jax.tree.map(np.testing.assert_array_equal, a, c)
        assert not np.array_equal(a[0].noisy[0], b[0].noisy[0])


def test_sampling_covers_every_geometry_depth_and_orientation():
    counts = np.zeros((512, 6), np.int32)
    transforms = np.zeros((512, 8), np.int32)
    for step in range(study.STEPS):
        row = study.select(step)
        for i, code in zip(row["indices"], row["transforms"], strict=True):
            counts[i, row["depth"] - 1] += 1
            transforms[i, code] += 1
    assert counts.min() > 0 and transforms.min() > 0
    assert counts.sum() == transforms.sum() == study.STEPS * 4
