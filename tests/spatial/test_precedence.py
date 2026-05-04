import numpy as np

from python.spatial import haas

FS = 44100


def _sine(freq=440.0, n=4096):
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * freq * t)


def test_haas_shape():
    sig = _sine()
    L, R = haas(sig, 100)
    assert L.shape == sig.shape
    assert R.shape == sig.shape


def test_haas_left_is_dry():
    sig = _sine()
    L, R = haas(sig, 100)
    np.testing.assert_array_equal(L, sig)


def test_haas_right_is_delayed():
    sig = _sine()
    d = 50
    L, R = haas(sig, d)
    np.testing.assert_allclose(R[d:], sig[:len(sig) - d], atol=1e-12)
    np.testing.assert_array_equal(R[:d], np.zeros(d))
