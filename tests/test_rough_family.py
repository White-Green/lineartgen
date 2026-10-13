import jax
import jax.numpy as jnp
import numpy as np
from augmented_family_objective import augmented_gradient
from cue_capacity import make_arm, prepare_inputs
from expanded_family_data import make_case
from linear_family_objective import linear_gradient
from local_recursive_fit import fresh_model
from rough_family_data import BASE, STRESS, augmentation, rough_source, stress_options
from stable_assignment_study import update
from test_family_objective import candidates
from test_path_profile import annotation
from test_plausible_lineart import fixture


def test_source_corruptions_preserve_clean_and_blank_and_base():
    data, _ = make_case("new_probe", 6, 20261013)
    original = data["variants"].copy()
    np.testing.assert_array_equal(rough_source(data["mixture"], BASE), data["source"])
    faint = rough_source(data["mixture"], STRESS["faint"])
    dark = rough_source(data["mixture"], STRESS["dark"])
    np.testing.assert_allclose(1 - dark, 4 * (1 - faint), atol=2e-7)
    for name in STRESS:
        opts = stress_options(name, 6)
        a = rough_source(data["mixture"], opts)
        assert a.dtype == np.float32 and np.isfinite(a).all() and a.min() >= -1 and a.max() <= 1
        np.testing.assert_array_equal(
            rough_source(np.ones_like(data["mixture"]), opts), np.ones_like(a)
        )
    np.testing.assert_array_equal(original, data["variants"])
    assert augmentation(17, 12, 3) == augmentation(17, 12, 3)
    values = [augmentation(17, s, 0) for s in range(100)]
    assert any(x == BASE for x in values) and any(x != BASE for x in values)


def test_fixed_objective_exact_and_augmented_actual_update():
    config, p, _ = fixture()
    _, _, v = candidates()
    ann = annotation()
    model = fresh_model(config)
    fixed = ann | dict(opacity=jnp.asarray([0.4]))
    jax.tree.map(
        np.testing.assert_array_equal,
        augmented_gradient()(model, p, v, fixed),
        linear_gradient("linear")(model, p, v, ann),
    )
    source = 1 - (1 - p.guides[0]) * 0.5
    prepared = prepare_inputs(v[:, 0], source, p.noise_levels[0], jax.random.PRNGKey(4), depth=2)
    arm = make_arm(config, model)
    arm["gradient"] = augmented_gradient()
    assert update(arm, prepared, v, ann | dict(opacity=jnp.asarray([0.2])))["gradient_norm"] > 0
