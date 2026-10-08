"""Check diagnostic data semantics independently of the production trainer."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from diagnose_learnability import generate_pairs, prepare_inputs  # noqa: E402

from lineartgen.config import ExperimentConfig  # noqa: E402
from lineartgen.diffusion import downsample  # noqa: E402


def test_pairs_repeat_exactly_and_probe_is_independent():
    train, source, records = generate_pairs(567, size=32)
    again, source_again, records_again = generate_pairs(567, size=32)
    probe, _, probe_records = generate_pairs(568, size=32)
    np.testing.assert_array_equal(train, again)
    np.testing.assert_array_equal(source, source_again)
    assert records == records_again and len(records) == 57
    assert records[0]["generation_key"] != probe_records[0]["generation_key"]
    assert not np.array_equal(train[:18], probe[:18])
    for i in (18, 37, 56):
        np.testing.assert_array_equal(source[i], 1)
    np.testing.assert_array_equal(train[:19], train[19:38])
    np.testing.assert_allclose(source[19:38], 1 - (1 - train[:19]) * 0.3)


def test_rough_guides_never_contain_teacher_and_targets_protect_source():
    config = ExperimentConfig.model_validate(
        {
            "model": {"source_conditioning": True},
            "matching": {"method": "independent"},
            "diffusion": {"pattern_weights": [0, 0, 1, 0], "identity_probability": 0},
        }
    )
    clean = jnp.ones((4, 32, 32, 1)).at[:, 16].set(-1)
    source = 1 - (1 - clean) * 0.3
    prepared = prepare_inputs(clean, source, jax.random.PRNGKey(20), config, 2)
    np.testing.assert_array_equal(prepared.guides[0], source)
    np.testing.assert_array_equal(prepared.guides[1], downsample(source))
    active = np.asarray(prepared.noise_levels[0]) > 0
    assert active.any() and (~active).any()
    np.testing.assert_array_equal(
        np.asarray(prepared.targets[0])[active], np.asarray(clean)[active]
    )
    np.testing.assert_array_equal(
        np.asarray(prepared.targets[0])[~active], np.asarray(source)[~active]
    )
    np.testing.assert_array_equal(prepared.targets[1], downsample(prepared.targets[0]))
    # Coarse noise is generated independently, not by reducing the noisy fine image.
    assert not np.array_equal(downsample(prepared.noisy[0]), prepared.noisy[1])
