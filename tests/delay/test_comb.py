import numpy as np
import pytest

from python.delay import comb_filter, feedback_delay
from python.generators import generate_impulse

FS = 44100


def _impulse(n=1024):
    _, imp = generate_impulse(fs=FS, duration=n / FS)
    return imp


def _sine(freq=440.0, n=1024):
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * freq * t)


def test_feedback_delay_shape():
    """feedback_delay must return an array of the same length as the input."""
    assert feedback_delay(_sine(), 100).shape == _sine().shape


def test_comb_filter_shape():
    """comb_filter must return an array of the same length as the input."""
    assert comb_filter(_sine(), 100).shape == _sine().shape


def test_feedback_delay_produces_echoes():
    """An impulse must produce echoes at integer multiples of the delay; each echo must be attenuated by feedback^n."""
    sig = _impulse(2048)
    d = 100
    fb = 0.5
    out = feedback_delay(sig, d, feedback=fb)
    assert abs(out[0]) == pytest.approx(1.0, abs=1e-6)
    assert abs(out[d]) == pytest.approx(fb, abs=1e-6)
    assert abs(out[2 * d]) == pytest.approx(fb ** 2, abs=1e-6)
    assert abs(out[3 * d]) == pytest.approx(fb ** 3, abs=1e-6)


def test_feedback_delay_zero_feedback_is_identity():
    """With feedback=0 no echo is added back, so the output must equal the input."""
    sig = _sine()
    np.testing.assert_allclose(feedback_delay(sig, 50, feedback=0.0), sig, atol=1e-12)


def test_comb_filter_adds_delayed_copy():
    """An impulse must appear at index 0 (direct) and at index d (delayed copy scaled by gain); nothing in between."""
    sig = _impulse()
    d = 100
    g = 0.5
    out = comb_filter(sig, d, gain=g)
    assert out[0] == pytest.approx(1.0)
    assert out[d] == pytest.approx(g)
    assert np.all(np.abs(out[1:d]) < 1e-12)


def test_comb_filter_zero_gain_is_identity():
    """With gain=0 no delayed copy is mixed in, so the output must equal the input."""
    sig = _sine()
    np.testing.assert_allclose(comb_filter(sig, 50, gain=0.0), sig, atol=1e-12)

