import jax.numpy as jnp
import numpy as np
from attraction_augmentation import points, spatial, target_action
from attraction_graph import decode, logits_from_target, target
from attraction_objective import stage_loss
from attraction_radius_objective import narrow_gradient, wide_gradient
from attraction_radius_study import augment_prepared
from flax import nnx
from group_field_model import INVERSE
from projected_edge_gate import gate

from lineartgen.diffusion import ProgressTrainingInputs


def test_narrow_oracle_all_levels_and_d4_recomputation():
    s = np.array([[[3.21, 5.37], [14.83, 13.19]], [[14.83, 13.19], [27.46, 7.82]]])
    w = np.full((2, 2), 2.0)
    for level in range(6):
        t, _ = target(s, w, np.empty((0, 4)), 32, level, base_radius=1.0)
        h = logits_from_target(t, level, base_radius=1.0)
        parts = stage_loss(
            jnp.asarray(h[None]), jnp.asarray(t[None]), level, sparse=True, base_radius=1.0
        )
        assert float(parts["position"]) < 1e-10 and float(parts["width"]) == 0
        gh, rejected = gate(h, level, base_radius=1.0)
        assert rejected == 0
        np.testing.assert_array_equal(gh[..., 4:8] >= 0, h[..., 4:8] >= 0)
    t, _ = target(s, w, np.empty((0, 4)), 32, base_radius=1.0)
    original, _ = decode(logits_from_target(t, base_radius=1.0), base_radius=1.0)
    for g in range(8):
        expected, _ = target(points(s, 32, g), w, np.empty((0, 4)), 32, base_radius=1.0)
        transformed = target_action(t, g)
        np.testing.assert_allclose(transformed, expected, atol=2e-6, rtol=2e-6)
        rendered, _ = decode(logits_from_target(transformed, base_radius=1.0), base_radius=1.0)
        np.testing.assert_allclose(rendered, spatial(original, g), atol=1e-5, rtol=1e-5)


def test_prepared_every_field_transforms_with_item_and_inverse():
    rng = np.random.default_rng(10)
    values = [
        tuple(jnp.asarray(rng.normal(size=(4, n, n, 1)).astype(np.float32)) for n in (16, 8, 4))
        for _ in range(5)
    ]
    p = ProgressTrainingInputs(*values)
    for start in (0, 4):
        groups = np.arange(start, start + 4)
        q = augment_prepared(p, groups)
        restored = augment_prepared(q, [INVERSE[g] for g in groups])
        for before, after, changed in zip(p, restored, q, strict=True):
            for a, b, c in zip(before, after, changed, strict=True):
                np.testing.assert_array_equal(a, b)
                for k, g in enumerate(groups):
                    np.testing.assert_array_equal(c[k], spatial(np.asarray(a[k]), int(g)))


def test_both_nnx_wrappers_execute_all_stage_gradients():
    class TinyHead(nnx.Module):
        def __init__(self):
            self.bias = nnx.Param(jnp.zeros(9))

        def __call__(self, image, noise, insert, guide, progress, **kwargs):
            return jnp.broadcast_to(self.bias[...], (*image.shape[:-1], 9))

    images = tuple(jnp.ones((1, n, n, 1)) for n in (16, 8))
    prepared = ProgressTrainingInputs(images, images, images, images, images)
    targets = tuple(jnp.zeros((*a.shape[:-1], 9)) for a in images)
    for gradient in (wide_gradient, narrow_gradient):
        parts, grads = gradient(TinyHead(), prepared, targets)
        assert all(np.isfinite(v) for v in parts.values())
        assert np.isfinite(grads["bias"][...]).all() and grads["bias"][0] > 0
