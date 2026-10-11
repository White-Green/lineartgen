import jax
import numpy as np
from aux_distance_fit import make_arms
from diagnose_learnability import update
from flax import nnx
from paper_band_fit import objective as original_objective
from test_distance_supervision import fixture

from lineartgen.config import save_config
from lineartgen.diffusion import recursive_forward_with_preclip
from lineartgen.models.serialization import load_inference, save_inference
from lineartgen.training import checkpoints, lineart_state
from lineartgen.training.auxiliary_distance import (
    AuxiliaryDistance,
    forward_with_auxiliary,
    objective,
)
from lineartgen.training.distance_supervision import distance_supervision


def build(root):
    core, config, prepared = fixture(root)
    for method in ("control", "auxiliary"):
        save_config(config, root / f"configs/{method}.json")
    return core, config, prepared


def test_zero_weight_preserves_all_core_predictions_loss_and_gradients(tmp_path):
    core, config, p = build(tmp_path)
    model = AuxiliaryDistance(core, config.model, rngs=nnx.Rngs(42))
    (actual, _), grad = nnx.value_and_grad(lambda m: objective(m, p, config, 0), has_aux=True)(
        model
    )
    (expected, _), expected_grad = nnx.value_and_grad(
        lambda m: original_objective(m, p, config), has_aux=True
    )(core)
    np.testing.assert_array_equal(actual, expected)
    jax.tree.map(np.testing.assert_array_equal, grad["core"], expected_grad)
    for leaf in jax.tree.leaves(grad["auxiliary"]):
        np.testing.assert_array_equal(leaf, 0)
    predictions, physical, _ = forward_with_auxiliary(model, p)
    expected = recursive_forward_with_preclip(core, p["noisy"], p["levels"], guides=p["guides"])
    jax.tree.map(np.testing.assert_array_equal, (predictions, physical), expected)


def test_auxiliary_parameters_never_enter_image_recursion_and_core_export(tmp_path):
    core, config, p = build(tmp_path)
    model = AuxiliaryDistance(core, config.model, rngs=nnx.Rngs(42))
    before = forward_with_auxiliary(model, p)
    for layer in model.auxiliary.layers:
        layer.kernel[...] = 3
        layer.bias[...] = -2
    after = forward_with_auxiliary(model, p)
    jax.tree.map(np.testing.assert_array_equal, before[:2], after[:2])
    assert not np.array_equal(before[2], after[2])
    save_inference(model.core, config, tmp_path / "export", provenance="core extraction test")
    restored, _ = load_inference(tmp_path / "export")
    actual = recursive_forward_with_preclip(restored, p["noisy"], p["levels"], guides=p["guides"])
    jax.tree.map(np.testing.assert_array_equal, after[:2], actual)
    assert "auxiliary" not in nnx.to_pure_dict(nnx.state(restored, nnx.Param))


def test_field_supervision_reaches_shared_encoder(tmp_path):
    core, config, p = build(tmp_path)
    model = AuxiliaryDistance(core, config.model, rngs=nnx.Rngs(42))
    model.auxiliary.layers[-1].kernel[...] = 0.01

    def field_loss(model):
        _, _, distance = forward_with_auxiliary(model, p)
        return distance_supervision(16 * distance, p["distance"], p["target"], p["levels"][0] > 0)

    value, grad = nnx.value_and_grad(field_loss)(model)
    assert value > 0
    assert sum(float(np.abs(x).sum()) for x in jax.tree.leaves(grad["core"]["encoders"])) > 0
    # Smaller-stage image heads can receive gradients through the recursive insert;
    # no assertion of zero head gradients would be valid for recursive depth2.
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(grad))


def test_whole_wrapper_adam_ema_restore_preserves_next_update(tmp_path):
    _, _, p = build(tmp_path)
    arms = make_arms(tmp_path)
    for arm in arms.values():
        assert update(arm, p)["gradient_norm"] > 0
    with checkpoints.manager(tmp_path / "aux-checkpoints") as saved:
        lineart_state.save(saved, arms, 1)
        restored = make_arms(tmp_path)
        assert lineart_state.restore(saved, restored) == 1
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 1),
        lineart_state.payload(restored, 1),
    )
    for method in arms:
        assert update(arms[method], p) == update(restored[method], p)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 2),
        lineart_state.payload(restored, 2),
    )
