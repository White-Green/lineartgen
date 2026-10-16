import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cue_capacity import prepare_inputs
from flax import nnx
from mixed_context_model import MixedContextFieldCNN
from mixed_context_model import load_field as old_load
from plausible_study import exact
from residual_axis_model import TAG, add_readout, infer, load_field, save_field
from test_group_field import cfg, random_head


def test_both_readouts_roundtrip_and_reject_legacy_loader(tmp_path):
    config = cfg(1)
    base = MixedContextFieldCNN(config.model, rngs=nnx.Rngs(15))
    random_head(base)
    source = jax.random.uniform(jax.random.PRNGKey(16), (1, 64, 64, 1)) * 2 - 1
    strength = jnp.full_like(source, 0.5).at[:, :16].set(0)
    p = prepare_inputs(source, source, strength, jax.random.PRNGKey(17), 3)
    for mode in ("mean", "choice"):
        model = add_readout(base, config, mode=mode)
        model.head.residual.kernel[...] = (
            jax.random.normal(jax.random.PRNGKey(18), model.head.residual.kernel.shape) * 0.1
        )
        expected = infer(model, p)
        save_field(model, config, tmp_path / mode, provenance="test")
        restored, cc = load_field(tmp_path / mode)
        assert cc == config and restored.research_model == TAG and restored.head.mode == mode
        exact(nnx.state(model, nnx.Param), nnx.state(restored, nnx.Param))
        np.testing.assert_array_equal(expected, infer(restored, p))
        with pytest.raises(AssertionError):
            old_load(tmp_path / mode)
