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
    assert delay_line(_sine(), 100).shape == _sine().shape


def test_delay_line_zero_delay_is_identity():
    sig = _sine()
    np.testing.assert_array_equal(delay_line(sig, 0), sig)


def test_delay_line_shifts_samples():
    sig = _impulse()
    d = 50
    out = delay_line(sig, d)
    assert out[d] == pytest.approx(1.0)
    assert np.all(out[:d] == 0.0)


def test_delay_line_fills_zeros_at_start():
    sig = np.ones(100)
    out = delay_line(sig, 10)
    np.testing.assert_array_equal(out[:10], np.zeros(10))
    np.testing.assert_array_equal(out[10:], np.ones(90))
