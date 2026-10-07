import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lineartgen.config import ExperimentConfig, save_config

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import prepare_ink_structure_confirmation as confirmation
from campaign_protocol import read, sha, write


def test_confirmation_uses_sealed_draws_and_preserves_partial_masks(tmp_path, monkeypatch):
    root = tmp_path / "study"
    root.mkdir()
    save_config(ExperimentConfig(), root / "config.json")
    # Small test fixture; never reads or generates the real confirmation set.
    plan = dict(
        geometry_kinds=["thin"],
        geometry_variants=[40],
        source_split="final",
        crop_size=512,
        source_count=1,
        seed=101,
        noise_seed_offset=100000,
        corruption_seed_offset=1000,
        families=["clean", "strokes"],
        severities={"clean": ["clean"], "strokes": ["strong"]},
        corruption_ranges={"strong": {}},
        strengths=[0.1, 0.5],
        partial_edits=[False, True],
        per_target_zero_strength_control=True,
        source_unseenness="test fixture only",
    )
    write(root / "confirmation-plan.json", plan)
    write(root / "dataset.json", {"test": True})
    model = tmp_path / "model"
    model.mkdir()
    for name in ["weights.npz", "metadata.json", "config.json"]:
        (model / name).write_bytes(b"fixture")
    selection = tmp_path / "seal.json"
    write(
        selection,
        dict(
            confirmation_plan_sha256=sha(root / "confirmation-plan.json"),
            selection_notes="fixture",
            models={
                "candidate": dict(
                    path=str(model),
                    files={p.name: sha(p) for p in model.iterdir()},
                )
            },
        ),
    )
    source = SimpleNamespace(name="fixture", variants=[(Path("unused"), 520, 520)])
    monkeypatch.setattr(confirmation, "KINDS", ("thin",))
    monkeypatch.setattr(confirmation, "discover", lambda config: [source])
    monkeypatch.setattr(confirmation, "split_experiment", lambda *args: ([], [], [source]))
    monkeypatch.setattr(confirmation, "dataset_description", lambda *args: {"test": True})
    gray = np.ones((512, 512, 1), np.float32)
    gray[:, 250:253] = 0
    monkeypatch.setattr(confirmation, "read_crop", lambda *args, **kwargs: gray)
    monkeypatch.setattr(
        confirmation,
        "line_case",
        lambda *args: (
            gray,
            np.asarray([[251, 250, 1, 0, 12, 0]], np.float32),
            np.asarray([[251, 10, 251, 500, 1]], np.float32),
        ),
    )
    monkeypatch.setattr(confirmation, "augment_sketch", lambda x, key, config: (x + 1) / 2)
    destination = tmp_path / "evaluation"
    confirmation.prepare(root, selection, destination)
    manifest = read(destination / "cases.json")
    assert len(manifest["cases"]) == 18  # two targets, eight conditions plus zero control
    assert len({c["seed"] for c in manifest["cases"]}) == 18
    partial = next(c for c in manifest["cases"] if c["family"] == "strokes" and c["partial"])
    with np.load(destination / partial["file"]) as arrays:
        assert arrays["image"].shape == arrays["source"].shape == (512, 512, 1)
        assert np.max(np.abs(arrays["source"] - arrays["image"])) == 1
        assert np.all(arrays["strength"][:102] == 0)
        assert np.all(arrays["strength"][:, :170] == 0)
        assert np.all(arrays["strength"][384:, 384:] == 0)
        assert arrays["strength"][250, 251, 0] == np.float32(0.1)
    assert confirmation.prepare(root, selection, destination) == destination
    # Existing confirmation inputs and selected checkpoints cannot silently change.
    first = destination / manifest["cases"][0]["file"]
    first.write_bytes(b"altered")
    with pytest.raises(AssertionError):
        confirmation.prepare(root, selection, destination)
    (model / "weights.npz").write_bytes(b"different checkpoint")
    with pytest.raises(AssertionError):
        confirmation.check_selection(selection, root / "confirmation-plan.json")
