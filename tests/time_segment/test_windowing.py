import numpy as np
import pytest

from python.time_segment import apply_window

FS = 44100


def _sine(n=4096):
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * 440.0 * t)


def test_apply_window_shape():
    sig = _sine()
    assert apply_window(sig).shape == sig.shape


def test_apply_window_hann_zeros_endpoints():
    sig = np.ones(1024)
    out = apply_window(sig, "hann")
    assert out[0] == pytest.approx(0.0, abs=1e-10)
    assert out[-1] == pytest.approx(0.0, abs=1e-10)


def test_apply_window_rectangular_is_identity():
    sig = _sine()
    np.testing.assert_array_equal(apply_window(sig, "rectangular"), sig)


def test_apply_window_reduces_amplitude():
    sig = np.ones(1024)
    out = apply_window(sig, "hann")
    assert np.max(out) < 1.0


def test_apply_window_unknown_raises():
    with pytest.raises(ValueError):
        apply_window(np.ones(64), "unknown_window")
