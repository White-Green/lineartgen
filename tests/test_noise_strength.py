import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pydantic import ValidationError

from lineartgen import diffusion
from lineartgen.config import DiffusionConfig, load_config
from lineartgen.training.comparison import comparison_configs


def test_strength_distribution_and_identity_frequency():
    config = DiffusionConfig(strength_sampling="low_mixture", pattern_weights=(1, 0, 0, 0))
    t = np.asarray(
        diffusion.noise_map(jnp.ones((131072, 1, 1, 1)), jax.random.PRNGKey(19), config)
    ).ravel()
    assert abs(np.mean(t == 0) - 0.1) < 0.004
    active = t[t > 0]
    assert active.min() >= 0.001 and active.max() < 1
    assert abs(np.mean(active <= 0.5) - 0.8) < 0.004
    assert abs(np.mean(active > 0.9) - 0.04) < 0.003
    assert abs(active.mean() - 0.3504) < 0.003
    q = np.asarray(diffusion.strength_from_uniform(jnp.array([0, 0.4, 0.8, 0.9, 1]), config))
    np.testing.assert_allclose(q, [0.001, 0.2505, 0.5, 0.75, 1], atol=1e-7)
    with pytest.raises(ValidationError):
        DiffusionConfig(strength_sampling="low_mixture", low_strength_max=0.2, min_noise=0.3)


def test_variants_change_only_strength_law_and_share_corruptions(monkeypatch):
    config = load_config("configs/noise-strength.toml")
    variants = comparison_configs(config, study="noise_strength")
    assert list(variants) == ["control", "low80"]
    assert variants["control"].model_copy(update={"diffusion": config.diffusion}) == config
    assert config.loss.boundary_weight == config.loss.cldice_weight == 0
    clean = jnp.ones((16, 64, 64, 1)).at[:, :, 27:30].set(-1)
    sources, normals = [], []
    augment, normal = diffusion.augment_sketch, jax.random.normal

    def capture_source(*args, **kwargs):
        value = augment(*args, **kwargs)
        sources.append(np.asarray(value))
        return value

    def capture_noise(*args, **kwargs):
        value = normal(*args, **kwargs)
        normals.append(np.asarray(value))
        return value

    monkeypatch.setattr(diffusion, "augment_sketch", capture_source)
    monkeypatch.setattr(jax.random, "normal", capture_noise)
    prepared, draws = [], []
    for settings in variants.values():
        normals.clear()
        prepared.append(
            diffusion.training_inputs(
                clean,
                jax.random.PRNGKey(22),
                settings.diffusion,
                settings.matching,
                levels=6,
                sketch=settings.sketch,
            )
        )
        draws.append(normals.copy())
    np.testing.assert_array_equal(sources[0], sources[1])
    assert len(draws[0]) == len(draws[1]) >= 6
    for a, b in zip(draws[0], draws[1], strict=True):
        np.testing.assert_array_equal(a, b)
    assert any(
        np.any(np.asarray(a) != np.asarray(b))
        for a, b in zip(prepared[0].noise_levels, prepared[1].noise_levels, strict=True)
    )
    for level in range(6):
        a, b = (x.noise_levels[level] for x in prepared)
        np.testing.assert_array_equal(a == 0, b == 0)
        assert np.all(np.asarray(b) <= np.asarray(a) + 1e-7)
        np.testing.assert_array_equal(prepared[0].targets[level], prepared[1].targets[level])
        for p in prepared:
            mask = np.asarray(p.noise_levels[level]) == 0
            np.testing.assert_array_equal(
                np.asarray(p.noisy[level])[mask], np.asarray(p.targets[level])[mask]
            )


def test_high_strength_cases_change_only_editable_strengths(tmp_path):
    import importlib.util
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path("scripts").resolve()))
    try:
        spec = importlib.util.spec_from_file_location(
            "strength_diagnostics_test", "scripts/strength_diagnostics.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    reference = tmp_path / "reference"
    reference.mkdir()
    image = np.arange(16, dtype=np.float32).reshape(4, 4, 1) / 16
    mask = np.full_like(image, 0.5)
    mask[:2] = 0
    for i in range(3):
        np.savez_compressed(
            reference / f"case-{i}.npz",
            image=image,
            source=image * 0.5,
            strength=mask,
            probes=np.ones((2, 4), np.float32),
        )
    cases = [
        dict(file=f"case-{i}.npz", strength=t, suite=suite, seed=17 + i, partial=True)
        for i, (t, suite) in enumerate(
            [(0.5, "sketch_geometry"), (0.1, "sketch_geometry"), (0.5, "legacy")]
        )
    ]
    (reference / "cases.json").write_text(json.dumps(dict(cases=cases)))
    output = tmp_path / "high-strength"
    rows = module.prepare(reference, output)
    assert len(rows) == 1 and rows[0]["seed"] == 17 and rows[0]["original_case"] == 0
    with np.load(output / rows[0]["file"]) as actual:
        np.testing.assert_array_equal(actual["image"], image)
        np.testing.assert_array_equal(actual["source"], image * 0.5)
        np.testing.assert_array_equal(actual["strength"], np.where(mask > 0, np.float32(0.9), 0))
        np.testing.assert_array_equal(actual["probes"], np.ones((2, 4), np.float32))
    with pytest.raises(FileExistsError):
        module.prepare(reference, output)


def test_sampling_audit_replays_training_keys_and_detects_changed_strength(tmp_path, monkeypatch):
    import json
    import shutil
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    from strength_diagnostics import audit_sampling

    shutil.copyfile("configs/noise-strength.toml", tmp_path / "config.toml")
    config = load_config(tmp_path / "config.toml")
    variants = comparison_configs(config, study="noise_strength")
    clean = jnp.ones((16, 32, 32, 1))
    rows = []
    for step in range(1, 4):
        # The training loop folds its zero-based update index into the key;
        # timings.jsonl stores the completed (one-based) update count.
        key = jax.random.fold_in(jax.random.PRNGKey(config.train.seed), step - 1)
        strengths = {}
        for name, settings in variants.items():
            prepared = diffusion.training_inputs(
                clean, key, settings.diffusion, settings.matching, levels=1
            )
            strengths[name] = np.asarray(prepared.noise_levels[0].max((1, 2, 3))).tolist()
        rows.append(dict(step=step, strengths=strengths))
    (tmp_path / "training").mkdir()
    path = tmp_path / "training/timings.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    result = audit_sampling(tmp_path, 3)
    assert result["shared_identity_exact"]
    assert all(row["examples"] == 48 for row in result["records"])
    rows[0]["strengths"]["low80"][0] += 0.01
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(AssertionError):
        audit_sampling(tmp_path, 3)
