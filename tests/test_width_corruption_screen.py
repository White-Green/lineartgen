import numpy as np
from campaign_protocol import read, write
from evaluate_width_corruption_interaction import METHODS, summarize


def test_black_interior_screen_uses_edited_pixels_and_separates_methods(tmp_path, monkeypatch):
    import evaluate_width_corruption_interaction as evaluator

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
                                halo_opacity_excess=0.0005 if method in ("combined",) else 0.001,
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
    predictions["combined"][:, :16] = 1
    for i in (717, 727):
        np.savez_compressed(out / f"predictions/case-{i:04d}.npz", **predictions)
    summarize(tmp_path)
    summary = read(out / "summary.json")
    assert summary["all_passed"] and all(summary["methods_passed"].values())
    predictions["combined"][:, 16:] = 0  # Edited black interior is now gray.
    np.savez_compressed(out / "predictions/case-0727.npz", **predictions)
    summarize(tmp_path)
    summary = read(out / "summary.json")
    assert summary["methods_passed"] == {"combined": False}
    assert len(summary["failures"]) == 1
    failure = summary["failures"][0]
    assert failure["guard"] == "black_interior_mean_at_least_095" and failure["case"] == 727

    # Auxiliary strength .3 is deliberately poor, but is outside frozen grouped gates.
    assert not any(row.get("strength") == 0.3 for row in summary["checks"])
    # Loss of thin-line recall fails the specific primary and preservation gates.
    for row in rows:
        if row["method"] == "combined" and row["domain"] == "geometry" and row["family"] == "blur":
            row["ink_f1_alpha50"] = 0.85
    write(out / "metrics.json", dict(rows=rows))
    summarize(tmp_path)
    summary = read(out / "summary.json")
    assert summary["methods_passed"] == {"combined": False}
    assert any(
        x["method"] == "combined" and x["guard"] == "geometry_blur_retention_ink_f1_alpha50"
        for x in summary["failures"]
    )


def test_factorial_interaction_and_descriptive_controls_do_not_add_gates(tmp_path, monkeypatch):
    import evaluate_width_corruption_interaction as evaluator

    out = tmp_path / "development"
    (out / "predictions").mkdir(parents=True)
    write(tmp_path / "reference.json", {})
    monkeypatch.setattr(evaluator, "check", lambda _: {})
    target = -np.ones((32, 32, 1), np.float32)
    monkeypatch.setattr(
        evaluator, "load_case", lambda *_: ({}, dict(image=target, strength=np.ones_like(target)))
    )
    f1 = {"initial": 0.99, "pixel": 0.99, "strong50": 0.90, "width6": 0.80, "combined": 0.99}
    rows = []
    for method in METHODS:
        for domain in ("geometry", "real"):
            for family in ("blur", "strokes"):
                rows.append(
                    dict(
                        method=method,
                        domain=domain,
                        family=family,
                        strength=0.5,
                        partial=False,
                        kind=domain,
                        ink_f1_alpha50=f1[method],
                        connection_cut_rate=0.0,
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
    for i in (717, 727):
        np.savez_compressed(out / f"predictions/case-{i:04d}.npz", **{m: target for m in METHODS})
    summarize(tmp_path)
    summary = read(out / "summary.json")
    assert summary["methods_passed"] == {"combined": True}
    assert {c["method"] for c in summary["checks"]} == {"combined"}
    interactions = read(out / "factorial-interaction.json")
    assert not interactions["independent_confirmation"]
    for row in interactions["rows"]:
        if row["family"] != "blank":
            np.testing.assert_allclose(row["difference_of_differences"]["ink_f1_alpha50"], 0.28)
