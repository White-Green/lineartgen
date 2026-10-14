import copy

import jax
import jax.numpy as jnp
import numpy as np
from cue_capacity import forward, prepare_inputs
from flax import nnx
from plausible_data import make_case
from style_family_data import styled_candidates
from test_full_source_normalization import config
from wide_style_data import allstroke_candidates

from lineartgen.config import ModelConfig
from lineartgen.diffusion import source_guides
from lineartgen.models import create_model


def test_wide_targets_become_thin_without_mutating_geometry():
    data, record = make_case("train", 58)
    original = copy.deepcopy(record)
    actual = allstroke_candidates(data, record)
    assert record == original
    np.testing.assert_array_equal(styled_candidates(data, record), data["variants"])
    assert actual.shape == data["variants"].shape
    assert ((1 - actual[13]) / 2).sum() < ((1 - data["variants"][13]) / 2).sum() * 0.2
    np.testing.assert_array_equal(actual[::3], actual[1::3])
    np.testing.assert_array_equal(actual[::3], actual[2::3])


def test_existing_other_stroke_and_fill_targets_stay_exact():
    for index in (0, 36, 54, 56):
        data, record = make_case("train", index)
        np.testing.assert_array_equal(
            allstroke_candidates(data, record), styled_candidates(data, record)
        )


def test_fresh_rf87_keeps_parameter_count_and_initial_function():
    c, p = config()
    mc = ModelConfig.model_validate(c.model.model_dump() | dict(local_dilations=(1, 4, 16)))
    plain, context = [create_model(x, rngs=nnx.Rngs(8)) for x in (c.model, mc)]
    jax.tree.map(
        np.testing.assert_array_equal, nnx.state(plain, nnx.Param), nnx.state(context, nnx.Param)
    )
    assert sum(x.size for x in jax.tree.leaves(nnx.state(context, nnx.Param))) == 15809
    jax.tree.map(np.testing.assert_array_equal, forward(plain, p), forward(context, p))


def test_wide_interior_and_fill_can_be_indistinguishable_to_rf15():
    c, _ = config()
    mc = ModelConfig.model_validate(c.model.model_dump() | dict(local_dilations=(1, 4, 16)))
    source = np.ones((2, 128, 128, 1), np.float32)
    source[0, 16:112, 48:80] = 0.2
    source[1, 16:112, 16:112] = 0.2
    p = prepare_inputs(source, source, np.full_like(source, 0.5), jax.random.PRNGKey(0), depth=1)
    plain, context = [create_model(x, rngs=nnx.Rngs(8)) for x in (c.model, mc)]
    g = np.asarray(source_guides(plain, p.guides)[0])
    np.testing.assert_array_equal(g[0, 57:72, 63:78], g[1, 57:72, 63:78])
    # x70 is paper after converting the 32px stroke centered at x63.5 to 2px,
    # but black inside the large filled rectangle. No RF15 function can give both.
    for m in (plain, context):
        m.head.kernel[...] = jnp.full_like(m.head.kernel[...], 0.1)
    a = np.asarray(forward(plain, p)[1][0])[:, 64, 70, 0]
    b = np.asarray(forward(context, p)[1][0])[:, 64, 70, 0]
    np.testing.assert_array_equal(a[0], a[1])
    assert abs(float(b[0] - b[1])) > 1e-6
