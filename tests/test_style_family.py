import jax
import jax.numpy as jnp
import numpy as np
from augmented_family_objective import linear_costs
from cue_capacity import forward, make_arm
from evaluate_plausible import metrics
from expanded_family_data import make_case
from local_recursive_fit import fresh_model
from path_profile_objective import path_costs
from plausible_data import CENTER
from style_family_data import extended_data, styled_candidates
from style_family_objective import style_gradient, style_path_costs, style_pixel_costs
from style_family_study import update
from test_family_objective import candidates
from test_path_profile import annotation
from test_plausible_lineart import fixture
from test_source_family import source_config


def test_fixed_style_keeps_geometry_and_taper_and_filled_controls():
    from copy import deepcopy

    for index in (0, 6, 18, 24, 30, 36, 42, 48, 54, 56, 58):
        data, record = make_case("new_probe", index, 20261013)
        original = data["variants"].copy()
        result = styled_candidates(data, record)
        np.testing.assert_array_equal(original, data["variants"])
        assert result.shape == original.shape and result.dtype == np.float32
        if record["kind"] in ("blank", "black", "wide"):
            np.testing.assert_array_equal(result, original)
        else:
            # Scaling original widths leaves canonical style unchanged.
            changed = deepcopy(record)
            for path in changed["metadata"]["path_geometry"]:
                path["widths"] = [x * 2 for x in path["widths"]]
            np.testing.assert_array_equal(result, styled_candidates(data, changed))
            np.testing.assert_array_equal(result[0], result[1])
            np.testing.assert_array_equal(result[1], result[2])
            extended = extended_data(data, record)
            values = metrics(
                result[CENTER],
                extended,
                np.full_like(data["source"], 0.5),
                record["kind"],
                "whole05",
            )
            assert values["family_profile_error"] == 0
            if values.get("connection_cut_rate") is not None:
                assert values["connection_cut_rate"] == 0
            if values.get("connection_bridge_rate") is not None:
                assert values["connection_bridge_rate"] == 0


def test_common_normalization_matches_existing_costs_and_actual_update():
    c, p, _ = fixture()
    _, _, v = candidates()
    ann = annotation()
    model = fresh_model(source_config(c))
    model.head.kernel[...] = jnp.full_like(model.head.kernel[...], 0.01)
    pred, raw = forward(model, p)
    np.testing.assert_array_equal(
        style_pixel_costs(raw, v, p.noise_levels, v), linear_costs(raw, v, p.noise_levels)
    )
    np.testing.assert_array_equal(
        style_path_costs(pred, v, p.noise_levels, ann, v), path_costs(pred, v, p.noise_levels, ann)
    )
    # Candidate selection must remain global and give a real trainable objective.
    arm = make_arm(source_config(c), model)
    arm["gradient"] = style_gradient()
    before = jax.tree.map(np.asarray, model.head.kernel[...]).copy()
    values = update(arm, p, v, ann | dict(normalizers=v))
    assert values["gradient_norm"] > 0 and not np.array_equal(
        before, np.asarray(arm["model"].head.kernel[...])
    )
    assert (
        abs(values["loss"] - values["family"] - 0.25 * values["profile"] - 0.05 * values["range"])
        < 1e-6
    )
