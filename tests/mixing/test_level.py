import numpy as np
import pytest

from python.mixing import gain, normalize

FS = 44100


def _sine(freq=440.0, n=4096):
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * freq * t)


def test_gain_shape():
    """gain must return an array of the same length as the input."""
    sig = _sine()
    assert gain(sig, 6.0).shape == sig.shape


def test_normalize_shape():
    """normalize must return an array of the same length as the input."""
    sig = _sine()
    assert normalize(sig).shape == sig.shape


def test_gain_0db_is_identity():
    """0 dB gain is a multiplication by exactly 1; the output must equal the input."""
    sig = _sine()
    np.testing.assert_allclose(gain(sig, 0.0), sig, atol=1e-12)


def test_gain_6db_doubles_amplitude():
    """+6.0206 dB corresponds to a linear gain of exactly 2; sample values must double."""
    sig = np.array([1.0, -1.0, 0.5])
    out = gain(sig, 6.0206)
    np.testing.assert_allclose(out, sig * 2.0, atol=1e-3)


def test_gain_negative_attenuates():
    """Negative gain_db must reduce amplitude below the input level."""
    sig = np.ones(100)
    out = gain(sig, -6.0)
    assert np.max(out) < 1.0


def test_gain_positive_amplifies():
    """Positive gain_db must increase amplitude above the input level."""
    sig = np.ones(100) * 0.5
    out = gain(sig, 6.0)
    assert np.max(out) > 0.5


def test_normalize_sets_peak():
    """normalize must scale the signal so the peak equals 10^(target_db/20)."""
    sig = _sine() * 0.1
    target = -3.0
    out = normalize(sig, target_db=target)
    expected_peak = 10 ** (target / 20)
    assert np.max(np.abs(out)) == pytest.approx(expected_peak, rel=1e-4)


def test_normalize_silent_returns_unchanged():
    """A silent signal has no defined peak; normalize must return it unchanged rather than dividing by zero."""
    sig = np.zeros(100)
    out = normalize(sig, target_db=-3.0)
    np.testing.assert_array_equal(out, sig)


def test_normalize_0db_peak():
    """target_db=0.0 must scale the signal so the peak equals exactly 1.0."""
    sig = _sine() * 0.5
    out = normalize(sig, target_db=0.0)
    assert np.max(np.abs(out)) == pytest.approx(1.0, rel=1e-4)


def test_normalize_twice_is_idempotent():
    """Applying normalize twice must return the same result as applying it once."""
    sig = _sine() * 0.3
    out1 = normalize(sig, target_db=-6.0)
    out2 = normalize(out1, target_db=-6.0)
    np.testing.assert_allclose(out1, out2, atol=1e-12)


def test_gain_silent_signal_returns_zeros():
    """gain applied to an all-zero signal must return all zeros (no NaN from 0 × 10^(g/20))."""
    sig = np.zeros(100)
    out = gain(sig, 12.0)
    np.testing.assert_array_equal(out, np.zeros(100))

