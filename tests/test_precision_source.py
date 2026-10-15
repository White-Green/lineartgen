import numpy as np
from precision_source_data import quant_choice, quantize_source


def test_eight_bit_storage_round_trip():
    source = np.linspace(-1, 1, 10001, dtype=np.float32).reshape(1, -1, 1)
    result = quantize_source(source)
    assert result.dtype == np.float32 and result.min() == -1 and result.max() == 1
    assert np.max(abs(result - source)) <= 1 / 255 + 1e-7
    np.testing.assert_array_equal(quantize_source(result), result)
    assert len(np.unique(result)) == 256


def test_storage_sampling_is_independent_and_mixed():
    choices = [quant_choice(2026101502, s, i) for s in range(6000) for i in range(4)]
    assert 0.48 < np.mean(choices) < 0.52
    assert choices == [quant_choice(2026101502, s, i) for s in range(6000) for i in range(4)]
