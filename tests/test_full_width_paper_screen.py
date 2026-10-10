from pathlib import Path

import numpy as np
from campaign_protocol import read, write
from evaluate_full_width_paper import METHODS, summarize


def test_black_interior_screen_uses_edited_pixels_and_separates_methods(tmp_path, monkeypatch):
    import evaluate_full_width_paper as evaluator

    out = tmp_path / "development"
    (out / "predictions").mkdir(parents=True)
    write(tmp_path / "reference.json", {})
    monkeypatch.setattr(evaluator, "check", lambda _: {})
    target = -np.ones((32, 32, 1), np.float32)
    strength = np.ones_like(target) * 0.5
    strength[:, :16] = 0
    monkeypatch.setattr(
        evaluator, "load_case", lambda *_: ({}, dict(image=target, strength=strength))
    )
    rows = []
    for method in METHODS:
        for domain in ("real", "geometry"):
            for family in ("clean", "tone", "blur", "strokes"):
                for t in (0.1, 0.3, 0.5):
                    for partial in (False, True):
                        rows.append(
                            dict(
                                method=method,
                                domain=domain,
                                family=family,
                                strength=t,
                                partial=partial,
                                kind=domain,
                                ink_f1_alpha50=0.0 if t == 0.3 else 0.99,
                                connection_cut_rate=0.0,
                                ink_deficit_fraction=0.02,
                                halo_opacity_excess=0.0005 if method in ("wide",) else 0.001,
                            )
                        )
        rows.append(
            dict(
                method=method,
                domain="geometry",
                family="blank",
                strength=0.5,
                partial=False,
                kind="blank",
                blank_max_opacity=0.0,
            )
        )
    write(out / "metrics.json", dict(rows=rows))
    predictions = {m: target.copy() for m in METHODS}
    # Arbitrary changes to protected pixels must not reduce edited-interior opacity.
    predictions["wide"][:, :16] = 1
    for i in (717, 727):
        np.savez_compressed(out / f"predictions/case-{i:04d}.npz", **predictions)
    summarize(tmp_path)
    summary = read(out / "summary.json")
    assert summary["all_passed"] and all(summary["methods_passed"].values())
    predictions["wide"][:, 16:] = 0  # Edited black interior is now gray.
    np.savez_compressed(out / "predictions/case-0727.npz", **predictions)
    summarize(tmp_path)
    summary = read(out / "summary.json")
    assert summary["methods_passed"] == {"wide": False}
    assert len(summary["failures"]) == 1
    failure = summary["failures"][0]
    assert failure["guard"] == "black_interior_mean_at_least_095" and failure["case"] == 727

    # Auxiliary strength .3 is deliberately poor, but is outside frozen grouped gates.
    assert not any(row.get("strength") == 0.3 for row in summary["checks"])
    # Loss of thin-line recall fails the specific primary and preservation gates.
    for row in rows:
        if row["method"] == "wide" and row["domain"] == "geometry" and row["family"] == "blur":
            row["ink_f1_alpha50"] = 0.85
    write(out / "metrics.json", dict(rows=rows))
    summarize(tmp_path)
    summary = read(out / "summary.json")
    assert summary["methods_passed"] == {"wide": False}
    assert any(
        x["method"] == "wide" and x["guard"] == "geometry_blur_retention_ink_f1_alpha50"
        for x in summary["failures"]
    )


def test_all_frozen_AE_checks_reproduce_exactly(tmp_path, monkeypatch):
    import evaluate_full_width_paper as evaluator

    parent = Path("/home/whitegreen/projects/lineartgen/.worktrees") / (
        "jax-paper-band-generalization-20261010/runs/paper-band-generalization-20261010"
    )
    out = tmp_path / "development"
    (out / "predictions").mkdir(parents=True)
    plan = read(parent / "plan.json")
    monkeypatch.setattr(evaluator, "check", lambda _: plan)
    write(tmp_path / "reference.json", read(parent / "reference.json"))
    rows = read(parent / "development/metrics.json")["rows"]
    rows = [
        dict(x, method={"clearance": "band", "band": "wide"}.get(x["method"], x["method"]))
        for x in rows
    ]
    write(out / "metrics.json", dict(rows=rows))
    for i in (717, 727):
        with np.load(parent / f"development/predictions/case-{i:04d}.npz") as old:
            arrays = {
                "initial": old["initial"],
                "pixel": old["pixel"],
                "band": old["clearance"],
                "wide": old["band"],
            }
        np.savez_compressed(out / f"predictions/case-{i:04d}.npz", **arrays)
    summarize(tmp_path)
    actual = read(out / "summary.json")
    expected = read(parent / "development/summary.json")
    assert len(actual["checks"]) == 229
    for key in ("checks", "failures", "all_passed", "methods_passed"):
        expected_value = expected[key]
        if key in ("checks", "failures"):
            expected_value = [dict(x, method="wide") for x in expected_value]
        elif key == "methods_passed":
            expected_value = {"wide": expected_value["band"]}
        assert actual[key] == expected_value
