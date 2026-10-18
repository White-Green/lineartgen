import jax
import jax.numpy as jnp
import numpy as np
from attraction_graph import logits_from_target, target
from attraction_objective import stage_loss


def test_correct_projected_geometry_has_zero_regression_loss_all_levels():
    for level in range(6):
        t, _ = target(
            np.array([[[4.2, 16.2], [27.7, 16.2]]]),
            np.full((1, 2), 2.0),
            np.empty((0, 4)),
            32,
            level,
        )
        h = jnp.asarray(logits_from_target(t, level)[None])
        for sparse in (False, True):
            p = stage_loss(h, jnp.asarray(t[None]), level, sparse=sparse)
            assert float(p["position"]) < 1e-10 and float(p["width"]) == 0


def test_attraction_gradient_moves_point_towards_line_and_ignores_background_geometry():
    t, _ = target(
        np.array([[[4.2, 16.2], [27.7, 16.2]]]), np.full((1, 2), 2.0), np.empty((0, 4)), 32
    )
    h = jnp.zeros((1, 32, 32, 9))
    for sparse in (False, True):
        grad = jax.grad(
            lambda a, sparse=sparse: sum(
                stage_loss(a, jnp.asarray(t[None]), 0, sparse=sparse).values()
            )
        )(h)
        assert np.isfinite(grad).all()
        assert grad[0, 14, 10, 2] < 0 and grad[0, 18, 10, 2] > 0
        assert np.all(np.asarray(grad[0, ..., 1:4])[t[..., 0] == 0] == 0)


def test_empty_images_remain_finite_with_background_suppression():
    h = jnp.zeros((1, 8, 8, 9))
    t = jnp.zeros_like(h)
    for sparse in (False, True):
        grad = jax.grad(lambda a, sparse=sparse: sum(stage_loss(a, t, 0, sparse=sparse).values()))(
            h
        )
        assert np.isfinite(grad).all() and np.all(grad[..., 0] > 0)


def test_nnx_gradient_wrappers_execute_with_auxiliary_metrics():
    from attraction_objective import balanced_gradient, sparse_gradient
    from flax import nnx

    from lineartgen.diffusion import ProgressTrainingInputs

    class TinyHead(nnx.Module):
        def __init__(self):
            self.bias = nnx.Param(jnp.zeros(9))

        def __call__(self, image, noise, insert, guide, progress, **kwargs):
            return jnp.broadcast_to(self.bias[...], (*image.shape[:-1], 9))

    images = tuple(jnp.ones((1, n, n, 1)) for n in (16, 8))
    prepared = ProgressTrainingInputs(images, images, images, images, images)
    targets = tuple(jnp.zeros((*a.shape[:-1], 9)) for a in images)
    for gradient in (balanced_gradient, sparse_gradient):
        parts, grads = gradient(TinyHead(), prepared, targets)
        assert all(np.isfinite(v) for v in parts.values())
        assert np.isfinite(grads["bias"][...]).all()
        assert grads["bias"][0] > 0
