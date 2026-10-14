import jax
import numpy as np
import optimized_geometry_study as study
from rough_family_data import BASE
from test_focused_geometry import data


def test_sampling_covers_each_shape_depth_and_orientation():
    depths = np.zeros((16, 6), np.int32)
    orientations = np.zeros((16, 8), np.int32)
    base = 0
    for step in range(study.STEPS):
        s = study.select(step)
        for i, g in zip(s["indices"], s["transforms"], strict=True):
            depths[i, s["depth"] - 1] += 1
            orientations[i, g] += 1
        base += sum(x == BASE for x in s["augmentation"])
    assert depths.min() > 0 and orientations.min() > 0
    assert depths.sum() == orientations.sum() == study.STEPS * 4
    assert 0.23 < base / depths.sum() < 0.27


def test_fresh_architecture_comparison_has_identical_augmented_batches(monkeypatch):
    options = [BASE, BASE | dict(opacity=0.2, sigma=2.4, amplitude=1.5)] * 2
    for depth in range(1, 7):
        monkeypatch.setattr(
            study,
            "select",
            lambda step, depth=depth: dict(
                indices=[0] * 4,
                depth=depth,
                augmentation=options,
                transforms=[0, 1, 4, 7],
            ),
        )
        a, b, c = (study.batch(data(), 0, m) for m in study.METHODS)
        jax.tree.map(np.testing.assert_array_equal, a, b)
        jax.tree.map(np.testing.assert_array_equal, a, c)
        assert len(a[0].guides) == depth


def test_optimized_initial_field_is_one_and_render_is_white_at_all_depths():
    import jax.numpy as jnp
    from cue_capacity import prepare_inputs
    from flax import nnx
    from subpixel_field_model import coverage, field_forward
    from test_group_field import cfg

    models = {m: study.initialize(cfg(study.WIDTHS[m]), m) for m in study.METHODS}
    source = jax.random.uniform(jax.random.PRNGKey(33), (1, 64, 64, 1)) * 2 - 1
    forward = nnx.jit(field_forward)
    for depth in range(1, 7):
        prepared = prepare_inputs(
            source, source, jnp.full_like(source, 0.5), jax.random.PRNGKey(34), depth
        )
        for model in models.values():
            for field in forward(model, prepared):
                np.testing.assert_array_equal(field, 1)
                np.testing.assert_array_equal(coverage(field), 1)
