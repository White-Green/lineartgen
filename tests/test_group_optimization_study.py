import group_optimization_study as study
import jax
import jax.numpy as jnp
import numpy as np
from cue_capacity import prepare_inputs
from flax import nnx
from rough_family_data import BASE
from subpixel_field_model import coverage, field_forward
from test_focused_geometry import data
from test_group_field import cfg


def test_factorial_initialization_changes_only_bias_and_lr():
    base = cfg()
    base = base.model_copy(update=dict(train=base.train.model_copy(update=dict(batch_size=4))))
    configs = {m: study.configured(base, m) for m in study.METHODS}
    models = {m: study.initialize(configs[m], m) for m in study.METHODS}
    first = nnx.to_pure_dict(nnx.state(models[study.METHODS[0]], nnx.Param))
    for m in study.METHODS:
        assert configs[m].train.peak_learning_rate == (1e-3 if m.endswith("fast") else 1e-4)
        state = nnx.to_pure_dict(nnx.state(models[m], nnx.Param))
        np.testing.assert_array_equal(state["head"]["linear"]["bias"], study.initial_field(m) / 8)
        state["head"]["linear"]["bias"] = first["head"]["linear"]["bias"]
        jax.tree.map(np.testing.assert_array_equal, state, first)
    source = jax.random.uniform(jax.random.PRNGKey(31), (1, 64, 64, 1)) * 2 - 1
    forward = nnx.jit(field_forward)
    for depth in range(1, 7):
        p = prepare_inputs(
            source, source, jnp.full_like(source, 0.5), jax.random.PRNGKey(32), depth
        )
        for m in study.METHODS:
            fields = forward(models[m], p)
            for field in fields:
                np.testing.assert_array_equal(field, study.initial_field(m))
                np.testing.assert_array_equal(coverage(field), 1)


def test_sampling_covers_all_shapes_depths_and_orientations():
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


def test_four_arms_receive_identical_augmented_inputs(monkeypatch):
    options = [BASE, BASE | dict(opacity=0.2, sigma=2.4, amplitude=1.5)] * 2
    for depth in range(1, 7):
        monkeypatch.setattr(
            study,
            "select",
            lambda step, depth=depth: dict(
                indices=[0] * 4, depth=depth, augmentation=options, transforms=[0, 1, 4, 7]
            ),
        )
        batches = [study.batch(data(), 0, m) for m in study.METHODS]
        for actual in batches[1:]:
            jax.tree.map(np.testing.assert_array_equal, batches[0], actual)
        assert len(batches[0][0].guides) == depth
