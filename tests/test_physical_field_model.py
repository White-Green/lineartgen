import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cue_capacity import prepare_inputs
from flax import nnx
from group_field_model import GroupFieldCNN
from physical_field_model import TAG, convert, infer, load_field, save_field
from plausible_study import exact
from subpixel_field_model import load_field as old_load
from test_group_field import cfg, random_head


def test_physical_serialization_preserves_nonzero_forward_and_rejects_legacy(tmp_path):
    config = cfg(1)
    old = GroupFieldCNN(config.model, rngs=nnx.Rngs(42))
    random_head(old)
    new = convert(old, config)
    exact(nnx.state(new, nnx.Param), nnx.state(old, nnx.Param))
    source = jax.random.uniform(jax.random.PRNGKey(1), (1, 64, 64, 1)) * 2 - 1
    p = prepare_inputs(source, source, jnp.full_like(source, 0.5), jax.random.PRNGKey(2), 6)
    expected = np.asarray(infer(new, p))
    assert np.ptp(expected) > 0  # Serialization must compare a nonconstant prediction.
    save_field(new, config, tmp_path / "model", provenance="test")
    restored, actual_config = load_field(tmp_path / "model")
    assert actual_config == config and restored.research_model == TAG
    exact(nnx.state(new, nnx.Param), nnx.state(restored, nnx.Param))
    np.testing.assert_array_equal(infer(restored, p), expected)
    with pytest.raises(AssertionError):
        old_load(tmp_path / "model")
