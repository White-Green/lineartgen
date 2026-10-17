import jax
import jax.numpy as jnp
import numpy as np
from pair_gradient_projection import combine


def test_conflicting_vectors_and_first_order_descent():
    a = {"x": jnp.array([1.0, 0.0])}
    b = {"x": jnp.array([-1.0, 1.0])}
    avg, ma = combine(a, b, project=False)
    pc, mp = combine(a, b, project=True)
    np.testing.assert_array_equal(avg["x"], [0, 0.5])
    np.testing.assert_array_equal(pc["x"], [0.25, 0.75])
    assert float(mp["combined_dot_focus"]) > 0 and float(mp["combined_dot_background"]) > 0
    assert float(ma["conflicting"]) == 1


def test_aligned_zero_and_opposite_gradients():
    for aa, bb, expected in [
        ([1.0, 2.0], [2.0, 4.0], [1.5, 3.0]),
        ([0.0, 0.0], [2.0, 4.0], [1.0, 2.0]),
        ([2.0, 0.0], [-3.0, 0.0], [0.0, 0.0]),
    ]:
        pc, m = combine({"x": jnp.array(aa)}, {"x": jnp.array(bb)}, project=True)
        # The 2/3 projection coefficient is rounded in float32.
        np.testing.assert_allclose(pc["x"], expected, atol=2e-7, rtol=1e-7)
        assert all(np.isfinite(float(x)) for x in m.values())


def test_tree_aggregation_matches_joint_mean_gradient():
    params = {"x": jnp.array([0.3, -0.2]), "b": jnp.array(0.1)}

    def f(p):
        return jnp.sum((p["x"] - 1) ** 2) + p["b"] ** 2

    def b(p):
        return jnp.sum((p["x"] + 2) ** 2) + 2 * p["b"] ** 2

    gf, gb = jax.grad(f)(params), jax.grad(b)(params)
    avg, _ = combine(gf, gb, project=False)
    joint = jax.grad(lambda p: (f(p) + b(p)) * 0.5)(params)
    for x, y in zip(jax.tree.leaves(avg), jax.tree.leaves(joint), strict=True):
        np.testing.assert_allclose(x, y, atol=1e-7, rtol=1e-7)
