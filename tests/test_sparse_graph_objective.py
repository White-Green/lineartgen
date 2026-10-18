import jax
import jax.numpy as jnp
import numpy as np
import optax
from learned_centerline_model import balanced_bce, stage_loss
from sparse_graph_objective import count_bce, sparse_stage_loss


def test_explicit_per_image_formula_and_empty():
    x = jnp.array([[[2.0, -3.0], [0.0, 1.0]], [[-4.0, 0.0], [2.0, 3.0]]])
    y = jnp.array([[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]])
    values = np.asarray(optax.sigmoid_binary_cross_entropy(x, y))
    expected = np.mean([values[0].sum() / (1 + 0.004 + 1e-6), values[1].sum() / (0.004 + 1e-6)])
    np.testing.assert_allclose(count_bce(x, y), expected, rtol=1e-6)
    assert np.isfinite(jax.grad(count_bce)(x, y)).all()


def test_background_not_diluted_and_positive_gradient_retained():
    x = jnp.zeros((1, 8, 8))
    y = x.at[0, 4, 4].set(1)
    a = jax.grad(count_bce)(x, y)
    b = jax.grad(balanced_bce)(x, y)
    assert a[0, 0, 0] > 50 * b[0, 0, 0] > 0
    assert a[0, 4, 4] < 0
    np.testing.assert_allclose(a[0, 4, 4] / b[0, 4, 4], 1 / (1 + 0.064 + 1e-6), rtol=1e-6)


def test_wrong_saturated_logits_have_nonzero_correct_gradient():
    x = jnp.array([[[80.0, -80.0]]])
    y = jnp.array([[[0.0, 1.0]]])
    g = jax.grad(count_bce)(x, y)
    assert g[0, 0, 0] > 0.9 and g[0, 0, 1] < -0.9


def test_only_vertex_edge_terms_change():
    h = jax.random.normal(jax.random.PRNGKey(1), (2, 8, 8, 9))
    t = jnp.zeros_like(h).at[:, 2:5, 3, :].set(1)
    a = stage_loss(h, t)
    b = sparse_stage_loss(h, t)
    for key in ("position", "width", "fill"):
        np.testing.assert_array_equal(a[key], b[key])
    for key in ("vertex", "edge"):
        assert a[key] != b[key]
