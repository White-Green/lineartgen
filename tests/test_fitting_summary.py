import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from complete_fitting_evaluation import fails_fit


def test_empty_active_ink_has_no_undefined_retention_comparison():
    row = dict(ink_f1_alpha50=1.0, ink_deficit_fraction=None, halo_opacity_excess=None)
    assert not fails_fit(row)
    assert fails_fit(dict(row, ink_f1_alpha50=0.5))
    assert fails_fit(dict(row, halo_opacity_excess=0.03))


def test_defined_fit_thresholds_are_unchanged():
    row = dict(ink_f1_alpha50=0.95, ink_deficit_fraction=0.05, halo_opacity_excess=0.02)
    assert not fails_fit(row)
    for name, value in [
        ("ink_f1_alpha50", 0.949),
        ("ink_deficit_fraction", 0.051),
        ("halo_opacity_excess", 0.021),
    ]:
        assert fails_fit(dict(row, **{name: value}))
