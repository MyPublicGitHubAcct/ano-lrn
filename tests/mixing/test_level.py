import numpy as np
import pytest

from python.mixing import gain, normalize

FS = 44100


def _sine(freq=440.0, n=4096):
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * freq * t)


def test_gain_shape():
    sig = _sine()
    assert gain(sig, 6.0).shape == sig.shape


def test_normalize_shape():
    sig = _sine()
    assert normalize(sig).shape == sig.shape


def test_gain_0db_is_identity():
    sig = _sine()
    np.testing.assert_allclose(gain(sig, 0.0), sig, atol=1e-12)


def test_gain_6db_doubles_amplitude():
    sig = np.array([1.0, -1.0, 0.5])
    out = gain(sig, 6.0206)
    np.testing.assert_allclose(out, sig * 2.0, atol=1e-3)


def test_gain_negative_attenuates():
    sig = np.ones(100)
    out = gain(sig, -6.0)
    assert np.max(out) < 1.0


def test_gain_positive_amplifies():
    sig = np.ones(100) * 0.5
    out = gain(sig, 6.0)
    assert np.max(out) > 0.5


def test_normalize_sets_peak():
    sig = _sine() * 0.1
    target = -3.0
    out = normalize(sig, target_db=target)
    expected_peak = 10 ** (target / 20)
    assert np.max(np.abs(out)) == pytest.approx(expected_peak, rel=1e-4)


def test_normalize_silent_returns_unchanged():
    sig = np.zeros(100)
    out = normalize(sig, target_db=-3.0)
    np.testing.assert_array_equal(out, sig)


def test_normalize_0db_peak():
    sig = _sine() * 0.5
    out = normalize(sig, target_db=0.0)
    assert np.max(np.abs(out)) == pytest.approx(1.0, rel=1e-4)
