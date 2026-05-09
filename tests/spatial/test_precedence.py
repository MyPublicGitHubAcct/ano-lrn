import numpy as np

from python.spatial import haas

FS = 44100


def _sine(freq=440.0, n=4096):
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * freq * t)


def test_haas_shape():
    """haas must return two arrays (L, R) each matching the input length."""
    sig = _sine()
    L, R = haas(sig, 100)
    assert L.shape == sig.shape
    assert R.shape == sig.shape


def test_haas_left_is_dry():
    """The Haas (precedence) effect delays only the right channel; the left channel must equal the input."""
    sig = _sine()
    L, R = haas(sig, 100)
    np.testing.assert_array_equal(L, sig)


def test_haas_right_is_delayed():
    """The right channel must be a d-sample delayed copy of the input, with d leading zeros."""
    sig = _sine()
    d = 50
    L, R = haas(sig, d)
    np.testing.assert_allclose(R[d:], sig[:len(sig) - d], atol=1e-12)
    np.testing.assert_array_equal(R[:d], np.zeros(d))


def test_haas_zero_delay_identical_channels():
    """delay=0 produces no precedence effect; left and right must be identical."""
    sig = _sine()
    L, R = haas(sig, 0)
    np.testing.assert_array_equal(L, R)

