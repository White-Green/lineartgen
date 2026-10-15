import jax
import jax.numpy as jnp
import numpy as np
from cue_capacity import prepare_inputs
from field_fit_study import gradient
from field_repair_study import objective
from flax import nnx
from physical_field_model import PhysicalGroupFieldCNN
from subpixel_field_model import field_forward
from test_group_field import cfg, random_head


def test_cnn_gradient_matches_direct_field_loss_when_context_is_kept():
    model = PhysicalGroupFieldCNN(cfg(1).model, rngs=nnx.Rngs(42))
    random_head(model)
    source = jax.random.uniform(jax.random.PRNGKey(9), (1, 16, 16, 1)) * 2 - 1
    p = prepare_inputs(source, source, jnp.full_like(source, 0.5), jax.random.PRNGKey(2), 2)
    fields = field_forward(model, p)
    targets = tuple(
        jnp.where(jnp.arange(f.shape[1])[None, None, :, None, None] % 8 < 4, -0.5, 1.0)
        * jnp.ones((1, 1, *f.shape[1:]))
        for f in fields
    )
    keep = jnp.ones((2, 1, 1, 1, 1))
    parts, actual = gradient(model, p, (targets, keep), None)

    def direct(m):
        return objective(field_forward(m, p), targets, p.noise_levels)

    (_, expected_parts), expected = nnx.value_and_grad(direct, has_aux=True)(model)
    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(a, b, atol=5e-6, rtol=1e-4),
        (parts, actual),
        (expected_parts, expected),
    )
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(actual))
    assert sum(float(np.linalg.norm(x)) for x in jax.tree.leaves(actual)) > 0
