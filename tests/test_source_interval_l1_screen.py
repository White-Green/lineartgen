import numpy as np
from campaign_protocol import read, write
from evaluate_source_interval_l1 import METHODS, summarize


def test_black_interior_screen_uses_edited_pixels_and_separates_methods(tmp_path, monkeypatch):
    import evaluate_source_interval_l1 as evaluator

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
                                halo_opacity_excess=0.0005 if method in ("interval",) else 0.001,
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
    predictions["interval"][:, :16] = 1
    for i in (717, 727):
        np.savez_compressed(out / f"predictions/case-{i:04d}.npz", **predictions)
    summarize(tmp_path)
    summary = read(out / "summary.json")
    assert summary["all_passed"] and all(summary["methods_passed"].values())
    predictions["interval"][:, 16:] = 0  # Edited black interior is now gray.
    np.savez_compressed(out / "predictions/case-0727.npz", **predictions)
    summarize(tmp_path)
    summary = read(out / "summary.json")
    assert summary["methods_passed"] == {"interval": False}
    assert len(summary["failures"]) == 1
    failure = summary["failures"][0]
    assert failure["guard"] == "black_interior_mean_at_least_095" and failure["case"] == 727

    # Auxiliary strength .3 is deliberately poor, but is outside frozen grouped gates.
    assert not any(row.get("strength") == 0.3 for row in summary["checks"])
    # Loss of thin-line recall fails the specific primary and preservation gates.
    for row in rows:
        if row["method"] == "interval" and row["domain"] == "geometry" and row["family"] == "blur":
            row["ink_f1_alpha50"] = 0.85
    write(out / "metrics.json", dict(rows=rows))
    summarize(tmp_path)
    summary = read(out / "summary.json")
    assert summary["methods_passed"] == {"interval": False}
    assert any(
        x["method"] == "interval" and x["guard"] == "geometry_blur_retention_ink_f1_alpha50"
        for x in summary["failures"]
    )
