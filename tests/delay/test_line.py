import numpy as np
import pytest

from python.delay import delay_line
from python.generators import generate_impulse

FS = 44100


def _impulse(n=1024):
    _, imp = generate_impulse(fs=FS, duration=n / FS)
    return imp


def _sine(freq=440.0, n=1024):
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * freq * t)


def test_delay_line_shape():
    """delay_line must return an array of the same length as the input."""
    assert delay_line(_sine(), 100).shape == _sine().shape


def test_delay_line_zero_delay_is_identity():
    """A zero-sample delay must return the signal unmodified — the trivial pass-through case."""
    sig = _sine()
    np.testing.assert_array_equal(delay_line(sig, 0), sig)


def test_delay_line_shifts_samples():
    """An impulse at index 0 must appear at index d after a d-sample delay, confirming the offset is exact."""
    sig = _impulse()
    d = 50
    out = delay_line(sig, d)
    assert out[d] == pytest.approx(1.0)
    assert np.all(out[:d] == 0.0)


def test_delay_line_fills_zeros_at_start():
    """The first d samples of the output must be zero (no wrap-around from the end of the signal)."""
    sig = np.ones(100)
    out = delay_line(sig, 10)
    np.testing.assert_array_equal(out[:10], np.zeros(10))
    np.testing.assert_array_equal(out[10:], np.ones(90))


def test_delay_line_delay_exceeds_length_is_all_zero():
    """When delay >= len(signal), the signal is pushed entirely beyond the buffer; output must be all-zero."""
    sig = _sine()
    out = delay_line(sig, len(sig))
    np.testing.assert_array_equal(out, np.zeros_like(sig))


def test_delay_line_fractional_delay_is_truncated():
    """Only integer delays are supported; a fractional value must be truncated, not rounded."""
    sig = _impulse()
    d_float = 50.7
    d_int = int(d_float)  # truncates to 50
    out = delay_line(sig, d_float)
    assert out[d_int] == pytest.approx(1.0)
    assert np.all(out[:d_int] == 0.0)

