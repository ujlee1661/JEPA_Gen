import math

import pytest
import torch

from jets.data import build_suffix_window
from jets.data.windows import _encode_suffix_spacing
from jets.layers.spacing import encode_spacing


def test_suffix_preserves_original_endpoint_without_rounding_it_down():
    end = 358.9992361111111
    anchor = 0.24024
    assert anchor + (end - anchor) < end  # Trigger the original float64 failure.
    window = {
        'count': 2,
        'anchor': 0.0,
        'delta': end,
        'times': torch.tensor([anchor, end], dtype=torch.float64),
        'marks': torch.zeros(2, 1),
        'attrs': torch.zeros(1),
        'past_tokens': torch.zeros(0, 2),
        'past_extension_tokens': torch.zeros(2, 2),
        'time_scale': 1.0,
    }
    result = build_suffix_window(window, 1)
    assert result['times'].tolist() == [end]
    assert result['gaps'].tolist() == [end - anchor, 0.0]
    assert result['active'].tolist() == [True, False]


def test_suffix_spacing_matches_existing_encoding_when_endpoint_is_representable():
    times = torch.tensor([1.0, 2.0, 2.0, 3.0], dtype=torch.float64)
    actual = _encode_suffix_spacing(times, 1.0, 3.0)
    expected = encode_spacing(times, 1.0, 2.0)
    for key in expected:
        assert torch.equal(actual[key], expected[key])


@pytest.mark.parametrize('times, anchor, end, message', [
    ([math.nan], 0.0, 1.0, 'Finite one-dimensional'),
    ([[0.5]], 0.0, 1.0, 'Finite one-dimensional'),
    ([1.000001], 0.0, 1.0, 'outside'),
    ([0.5, 0.4], 0.0, 1.0, 'ordered'),
    ([0.5], 1.0, 0.0, 'nonnegative horizon'),
    ([0.5], math.nan, 1.0, 'Finite anchor'),
])
def test_suffix_spacing_preserves_invalid_input_rejection(times, anchor, end, message):
    with pytest.raises(ValueError, match=message):
        _encode_suffix_spacing(torch.tensor(times, dtype=torch.float64), anchor, end)
