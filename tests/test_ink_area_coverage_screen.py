import numpy as np
from campaign_protocol import read, write
from evaluate_ink_area_coverage import METHODS, summarize


def test_black_interior_screen_uses_edited_pixels_and_separates_methods(tmp_path, monkeypatch):
    import evaluate_ink_area_coverage as evaluator

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
                for t in (0.1, 0.5):
                    for partial in (False, True):
                        rows.append(
                            dict(
                                method=method,
                                domain=domain,
                                family=family,
                                strength=t,
                                partial=partial,
                                kind=domain,
                                ink_f1_alpha50=0.99,
                                ink_deficit_fraction=0.02,
                                halo_opacity_excess=0.001,
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
    predictions["flat_coverage"][:, :16] = 1
    for i in (717, 727):
        np.savez_compressed(out / f"predictions/case-{i:04d}.npz", **predictions)
    summarize(tmp_path)
    summary = read(out / "summary.json")
    assert summary["all_passed"] and all(summary["methods_passed"].values())
    predictions["flat_coverage"][:, 16:] = 0  # Edited black interior is now gray.
    np.savez_compressed(out / "predictions/case-0727.npz", **predictions)
    summarize(tmp_path)
    summary = read(out / "summary.json")
    assert summary["methods_passed"] == {"coverage": True, "flat_coverage": False}
    assert len(summary["failures"]) == 1
    failure = summary["failures"][0]
    assert failure["guard"] == "black_interior_mean_at_least_095" and failure["case"] == 727
