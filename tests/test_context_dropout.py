"""Dropping an edge blocks coarse influence without disabling coarse supervision."""

import jax
import jax.numpy as jnp
import numpy as np
from context_dropout_objective import context_mask, dropout_forward
from cue_capacity import prepare_inputs
from flax import nnx
from group_field_model import GroupFieldCNN
from subpixel_field_model import field_forward
from test_group_field import cfg, random_head


def test_dropped_edge_blocks_fine_gradient_but_preserves_coarse_gradient():
    config = cfg(2).model
    model = GroupFieldCNN(config, rngs=nnx.Rngs(3))
    random_head(model)
    source = jax.random.uniform(jax.random.PRNGKey(6), (1, 16, 16, 1))
    p = prepare_inputs(source, source, jnp.full_like(source, 0.5), jax.random.PRNGKey(4), 2)
    keep = jnp.ones((2, 1, 1, 1, 1))
    a, b = dropout_forward(model, p, keep), field_forward(model, p)
    jax.tree.map(np.testing.assert_array_equal, a, b)
    drop = keep.at[1].set(0)

    def loss(coarse, mask, level):
        prepared = p._replace(noise_levels=(p.noise_levels[0], coarse))
        return dropout_forward(model, prepared, mask)[level].sum()

    fine_full = jax.grad(loss)(p.noise_levels[1], keep, 0)
    fine_dropped = jax.grad(loss)(p.noise_levels[1], drop, 0)
    coarse_dropped = jax.grad(loss)(p.noise_levels[1], drop, 1)
    assert np.max(abs(fine_full)) > 1e-5
    np.testing.assert_array_equal(fine_dropped, 0)
    assert np.max(abs(coarse_dropped)) > 1e-5


def test_masks_replay_and_have_no_spatial_or_level_zero_dropout():
    masks = []
    for step in range(60):
        depth = step % 6 + 1
        a = context_mask(step, depth, 4, enabled=True)
        np.testing.assert_array_equal(a, context_mask(step, depth, 4, enabled=True))
        np.testing.assert_array_equal(a[0], 1)
        np.testing.assert_array_equal(context_mask(step, depth, 4, enabled=False), 1)
        assert a.shape == (depth, 4, 1, 1, 1)
        masks.extend(a[1:].ravel())
    assert 0.4 < np.mean(masks) < 0.6
