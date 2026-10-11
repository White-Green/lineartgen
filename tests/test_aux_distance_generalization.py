import jax
import numpy as np
from aux_distance_generalization import prepared_batch, teachers
from flax import nnx
from paper_band_fit import objective as original_objective
from test_distance_supervision import fixture
from train_procedural_structure import prepare_batch

from lineartgen.training.auxiliary_distance import AuxiliaryDistance, objective
from lineartgen.training.coverage_distance import coverage_distance


def test_wrapper_preserves_runtime_source_and_one_pass_dispatch(tmp_path):
    from lineartgen.evaluation_sampling import inference_with_source

    core, config, p = fixture(tmp_path)
    model = AuxiliaryDistance(core, config.model, rngs=nnx.Rngs(7))
    for steps in (1, 5):
        expected = inference_with_source(core, p["noisy"], p["levels"], p["guides"][0], steps=steps)
        actual = inference_with_source(model, p["noisy"], p["levels"], p["guides"][0], steps=steps)
        np.testing.assert_array_equal(actual, expected)


def test_procedural_stream_is_complete_deterministic_and_step_keyed(tmp_path):
    _, config, _ = fixture(tmp_path)
    config = config.model_copy(
        update={
            "geometry": config.geometry.model_copy(
                update=dict(
                    probability=0.9, coverage_probability=1.0, minimum_width=0.6, maximum_width=6.0
                )
            )
        }
    )
    kinds, hashes = set(), set()
    for step in range(30):
        a, records = teachers(config, step, 64)
        b, repeated = teachers(config, step, 64)
        np.testing.assert_array_equal(a, b)
        assert records == repeated and len(records) == len(a)
        hashes.add(a.tobytes())
        kinds.update(k for r in records for k in r["kinds"])
    assert len(hashes) > 25 and len(kinds) >= 9


def test_shared_batch_preserves_clean_field_and_masked_image_targets(tmp_path):
    _, config, _ = fixture(tmp_path)
    config = config.model_copy(
        update={
            "geometry": config.geometry.model_copy(
                update=dict(probability=0.9, coverage_probability=1.0)
            ),
            "sketch": config.sketch.model_copy(
                update=dict(mode="mixed", probability=0.8, strong_probability=0.5)
            ),
            "matching": config.matching.model_copy(update={"method": "independent"}),
        }
    )
    prepare = prepare_batch(config)
    for step in range(3):
        clean, _ = teachers(config, step, 64)
        p, row = prepared_batch(config, prepare, step, 64)
        again, again_row = prepared_batch(config, prepare, step, 64)
        jax.tree.map(np.testing.assert_array_equal, p, again)
        assert row == again_row and row["depth"] == step + 1
        np.testing.assert_array_equal(p["field_clean"], clean)
        np.testing.assert_array_equal(
            p["distance"], np.stack([coverage_distance(y) for y in clean])
        )


def test_full_clean_field_labels_do_not_replace_main_protected_targets(tmp_path):
    core, config, p = fixture(tmp_path)
    p["field_clean"] = -p["target"]
    p["levels"] = tuple(x.at[:, :, :8].set(0) for x in p["levels"])
    model = AuxiliaryDistance(core, config.model, rngs=nnx.Rngs(7))
    (actual, _), grad = nnx.value_and_grad(lambda m: objective(m, p, config, 0), has_aux=True)(
        model
    )
    (expected, _), original_grad = nnx.value_and_grad(
        lambda m: original_objective(m, p, config), has_aux=True
    )(core)
    np.testing.assert_array_equal(actual, expected)
    jax.tree.map(np.testing.assert_array_equal, grad["core"], original_grad)
