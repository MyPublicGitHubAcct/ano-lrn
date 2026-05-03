import numpy as np
import pytest

from python.delay import comb_filter, delay_line, feedback_delay
from python.generators import generate_impulse, generate_sine

FS = 44100
DURATION = 0.1


def _impulse(n=1024):
    _, imp = generate_impulse(fs=FS, duration=n / FS)
    return imp


def _sine(freq=440.0, n=1024):
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * freq * t)


# ── Shape ─────────────────────────────────────────────────────────────────────

def test_delay_line_shape():
    sig = _sine()
    assert delay_line(sig, 100).shape == sig.shape


def test_feedback_delay_shape():
    sig = _sine()
    assert feedback_delay(sig, 100).shape == sig.shape


def test_comb_filter_shape():
    sig = _sine()
    assert comb_filter(sig, 100).shape == sig.shape


# ── Correctness ───────────────────────────────────────────────────────────────

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


def test_feedback_delay_produces_echoes():
    sig = _impulse(2048)
    d = 100
    fb = 0.5
    out = feedback_delay(sig, d, feedback=fb)
    # Impulse at 0, echoes at d, 2d, 3d decay by fb each hop.
    assert abs(out[0]) == pytest.approx(1.0, abs=1e-6)
    assert abs(out[d]) == pytest.approx(fb, abs=1e-6)
    assert abs(out[2 * d]) == pytest.approx(fb ** 2, abs=1e-6)
    assert abs(out[3 * d]) == pytest.approx(fb ** 3, abs=1e-6)


def test_feedback_delay_zero_feedback_is_identity():
    # H(z) = 1 / (1 - 0) = 1 when feedback = 0.
    sig = _sine()
    np.testing.assert_allclose(feedback_delay(sig, 50, feedback=0.0), sig, atol=1e-12)


def test_comb_filter_adds_delayed_copy():
    sig = _impulse(1024)
    d = 100
    g = 0.5
    out = comb_filter(sig, d, gain=g)
    # Impulse response: 1 at 0, g at d
    assert out[0] == pytest.approx(1.0)
    assert out[d] == pytest.approx(g)
    assert np.all(np.abs(out[1:d]) < 1e-12)


def test_comb_filter_zero_gain_is_identity():
    sig = _sine()
    np.testing.assert_allclose(comb_filter(sig, 50, gain=0.0), sig, atol=1e-12)
