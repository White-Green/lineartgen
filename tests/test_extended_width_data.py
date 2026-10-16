import copy

import numpy as np
from continuous_width_data import width_source
from extended_width_data import extended_width
from rough_family_data import BASE


def test_fixed_sampler_preserves_nonwide_and_native_choices():
    for step in range(100):
        assert extended_width(17, step, 0, 463, 12.0) == 12.0
        assert extended_width(17, step, 0, 464, None) is None
    values = [extended_width(17, k, 0, 464, 12.0) for k in range(1000)]
    assert values == [extended_width(17, k, 0, 464, 12.0) for k in range(1000)]
    changed = [x for x in values if x != 12.0]
    assert 400 < len(changed) < 600
    assert min(changed) >= 32 and max(changed) <= 64
    secondary = [extended_width(17, k, 0, 464, 12.0, component=1) for k in range(1000)]
    assert secondary != values


def test_source_widening_keeps_metadata_and_increases_ink_extent():
    points = np.column_stack((np.linspace(35, 155, 129), np.full(129, 64))).tolist()
    record = dict(
        kind="wide",
        direction=[0, 1],
        metadata=dict(path_geometry=[dict(kind="wide", points=points, widths=[20.0] * 129)]),
    )
    saved = copy.deepcopy(record)
    a = width_source({}, record, 464, 16.0, BASE)
    b = width_source({}, record, 464, 48.0, BASE)
    assert record == saved
    assert a.shape == b.shape == (256, 256, 1)
    assert np.isfinite(b).all() and b.min() >= -1 and b.max() <= 1
    assert float((1 - b).sum()) > 2.5 * float((1 - a).sum())
    np.testing.assert_array_equal(a, width_source({}, record, 464, 16.0, BASE))
