import numpy as np

from python.spatial import pan, stereo_widen

FS = 44100


def _sine(freq=440.0, n=4096):
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * freq * t)


def test_pan_shape():
    """pan must return two arrays (L, R) each matching the input length."""
    sig = _sine()
    L, R = pan(sig, 0.0)
    assert L.shape == sig.shape
    assert R.shape == sig.shape


def test_stereo_widen_shape():
    """stereo_widen must return two arrays (L, R) each matching the input length."""
    sig = _sine()
    L, R = stereo_widen(sig, sig)
    assert L.shape == sig.shape
    assert R.shape == sig.shape


def test_pan_centre_equal_power():
    """At position 0 (centre), L and R must be identical (equal power panning)."""
    sig = np.ones(1024)
    L, R = pan(sig, 0.0)
    np.testing.assert_allclose(L, R, atol=1e-12)


def test_pan_hard_left():
    """At position −1 (full left), all signal must appear in L and R must be silent."""
    sig = np.ones(1024)
    L, R = pan(sig, -1.0)
    np.testing.assert_allclose(L, sig, atol=1e-10)
    np.testing.assert_allclose(R, np.zeros_like(sig), atol=1e-10)


def test_pan_hard_right():
    """At position +1 (full right), all signal must appear in R and L must be silent."""
    sig = np.ones(1024)
    L, R = pan(sig, 1.0)
    np.testing.assert_allclose(L, np.zeros_like(sig), atol=1e-10)
    np.testing.assert_allclose(R, sig, atol=1e-10)


def test_pan_equal_power_law():
    """L² + R² must equal the input power at every pan position (constant-power panning law)."""
    sig = np.ones(1)
    for pos in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        L, R = pan(sig, pos)
        np.testing.assert_allclose(L ** 2 + R ** 2, sig ** 2, atol=1e-12)


def test_stereo_widen_width_0_is_mono():
    """width=0 collapses the stereo field to mono; both output channels must equal (L + R) / 2."""
    L = _sine(440.0)
    R = _sine(880.0)
    out_L, out_R = stereo_widen(L, R, width=0.0)
    expected = (L + R) * 0.5
    np.testing.assert_allclose(out_L, expected, atol=1e-12)
    np.testing.assert_allclose(out_R, expected, atol=1e-12)


def test_stereo_widen_width_1_is_identity():
    """width=1 applies no processing; both output channels must equal their respective inputs."""
    L = _sine(440.0)
    R = _sine(880.0)
    out_L, out_R = stereo_widen(L, R, width=1.0)
    np.testing.assert_allclose(out_L, L, atol=1e-12)
    np.testing.assert_allclose(out_R, R, atol=1e-12)


def test_stereo_widen_width_greater_than_one_increases_side():
    """width > 1 must increase the side (L − R) signal RMS compared to unprocessed stereo."""
    L = _sine(440.0)
    R = _sine(880.0)
    side_original = float(np.sqrt(np.mean((L - R) ** 2)))
    _, out_R = stereo_widen(L, R, width=2.0)
    out_L, _ = stereo_widen(L, R, width=2.0)
    side_widened = float(np.sqrt(np.mean((out_L - out_R) ** 2)))
    assert side_widened > side_original

