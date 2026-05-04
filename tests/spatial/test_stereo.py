import numpy as np

from python.spatial import pan, stereo_widen

FS = 44100


def _sine(freq=440.0, n=4096):
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * freq * t)


def test_pan_shape():
    sig = _sine()
    L, R = pan(sig, 0.0)
    assert L.shape == sig.shape
    assert R.shape == sig.shape


def test_stereo_widen_shape():
    sig = _sine()
    L, R = stereo_widen(sig, sig)
    assert L.shape == sig.shape
    assert R.shape == sig.shape


def test_pan_centre_equal_power():
    sig = np.ones(1024)
    L, R = pan(sig, 0.0)
    np.testing.assert_allclose(L, R, atol=1e-12)


def test_pan_hard_left():
    sig = np.ones(1024)
    L, R = pan(sig, -1.0)
    np.testing.assert_allclose(L, sig, atol=1e-10)
    np.testing.assert_allclose(R, np.zeros_like(sig), atol=1e-10)


def test_pan_hard_right():
    sig = np.ones(1024)
    L, R = pan(sig, 1.0)
    np.testing.assert_allclose(L, np.zeros_like(sig), atol=1e-10)
    np.testing.assert_allclose(R, sig, atol=1e-10)


def test_pan_equal_power_law():
    sig = np.ones(1)
    for pos in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        L, R = pan(sig, pos)
        np.testing.assert_allclose(L ** 2 + R ** 2, sig ** 2, atol=1e-12)


def test_stereo_widen_width_0_is_mono():
    L = _sine(440.0)
    R = _sine(880.0)
    out_L, out_R = stereo_widen(L, R, width=0.0)
    expected = (L + R) * 0.5
    np.testing.assert_allclose(out_L, expected, atol=1e-12)
    np.testing.assert_allclose(out_R, expected, atol=1e-12)


def test_stereo_widen_width_1_is_identity():
    L = _sine(440.0)
    R = _sine(880.0)
    out_L, out_R = stereo_widen(L, R, width=1.0)
    np.testing.assert_allclose(out_L, L, atol=1e-12)
    np.testing.assert_allclose(out_R, R, atol=1e-12)
