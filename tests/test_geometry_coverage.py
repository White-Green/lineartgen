import geometry_coverage_study as study
import jax
import jax.numpy as jnp
import numpy as np
from d4_geometry_data import transform_numpy_batch
from rough_family_data import BASE
from subpixel_field_data import geometry_field
from subpixel_field_model import coverage


def fixture():
    a = jnp.array([[[5.0, 16.0], [27.0, 16.0]]])
    b = jnp.array([[[16.0, 4.0], [16.0, 28.0]]])
    widths = jnp.full((1, 2), 2.0)
    rects = jnp.empty((0, 4))
    fields = np.stack([np.asarray(geometry_field(s, widths, rects, size=32)) for s in [a, b]])
    images = np.asarray(coverage(jnp.asarray(fields[:, 16:-16, 16:-16, None])))
    variants = np.repeat(images[:, None], 27, 1)
    sources = 1 - (1 - images) * 0.4
    return (
        dict(
            source=sources,
            variants=variants,
            fields=fields,
            directions=np.array([[0, 1], [1, 0]], np.float32),
        ),
        a,
        b,
        widths,
        rects,
    )


def selection(depth):
    return dict(
        indices=[0] * 4,
        companions=[1] * 4,
        compose=[True, False, True, False],
        modes=[0, 1, 2, 0],
        depth=depth,
        transforms=[0, 1, 4, 7],
        augmentation=[BASE] * 4,
    )


def test_composite_is_opaque_geometry_union_not_gray_average():
    data, a, b, widths, rects = fixture()
    original = {k: v.copy() for k, v in data.items()}
    source, clean, fields = study.components(data, selection(1), "composed")
    expected = geometry_field(
        jnp.concatenate([a, b]), jnp.concatenate([widths] * 2), rects, size=32
    )
    np.testing.assert_array_equal(fields[0], expected)
    rendered = np.asarray(coverage(jnp.asarray(fields[0:1, 16:-16, 16:-16, None])))[0]
    np.testing.assert_array_equal(rendered, clean[0])
    assert rendered[16, 16, 0] == -1 and rendered[16, 8, 0] == -1
    np.testing.assert_array_equal(source[0], 1 - (1 - clean[0]) * 0.4)
    for k in data:
        np.testing.assert_array_equal(data[k], original[k])
    for g in range(8):
        transformed = transform_numpy_batch(fields[0:1, ..., None], [g])
        parts = transform_numpy_batch(data["fields"][..., None], [g, g])
        np.testing.assert_array_equal(transformed[0], np.minimum(parts[0], parts[1]))


def test_shared_masks_rng_and_noncomposed_examples_all_depths(monkeypatch):
    data, *_ = fixture()
    original = study.strength_map
    monkeypatch.setattr(study, "strength_map", lambda mode: original(mode)[::8, ::8])
    for depth in range(1, 7):
        monkeypatch.setattr(study, "select", lambda step, depth=depth: selection(depth))
        a, b = [study.batch(data, 0, m) for m in study.METHODS]
        jax.tree.map(np.testing.assert_array_equal, a[0].noise_levels, b[0].noise_levels)
        np.testing.assert_array_equal(a[2], b[2])
        jax.tree.map(
            lambda x, y: np.testing.assert_array_equal(
                np.asarray(x)[[1, 3]], np.asarray(y)[[1, 3]]
            ),
            a,
            b,
        )
        assert np.any(np.asarray(a[0].guides[0]) != np.asarray(b[0].guides[0]))
        for pa, pb, ga, gb in zip(a[0].noisy, b[0].noisy, a[0].guides, b[0].guides, strict=True):
            np.testing.assert_allclose(pa - ga, pb - gb, atol=1.2e-7, rtol=0)


def test_sampling_keeps_blank_fill_and_wide_unchanged():
    composed = 0
    kinds = set()
    for step in range(study.STEPS):
        s = study.select(step)
        assert s == study.select(step)
        for i, j, active in zip(s["indices"], s["companions"], s["compose"], strict=True):
            assert 0 <= j < 432
            if active:
                assert i < 432
                composed += 1
                kinds.add((i // 48, j // 48))
    assert 0.40 < composed / (study.STEPS * 4) < 0.44
    assert len(kinds) == 81
