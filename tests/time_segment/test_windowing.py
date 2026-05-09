import numpy as np
import pytest

from python.time_segment import apply_window

FS = 44100


def _sine(n=4096):
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * 440.0 * t)


def test_apply_window_shape():
    """apply_window must return an array of the same length as the input."""
    sig = _sine()
    assert apply_window(sig).shape == sig.shape


def test_apply_window_hann_zeros_endpoints():
    """The Hann window is zero at index 0 and −1; windowed output at those positions must be zero."""
    sig = np.ones(1024)
    out = apply_window(sig, "hann")
    assert out[0] == pytest.approx(0.0, abs=1e-10)
    assert out[-1] == pytest.approx(0.0, abs=1e-10)


def test_apply_window_rectangular_is_identity():
    """A rectangular window multiplies by 1 everywhere; the output must equal the input exactly."""
    sig = _sine()
    np.testing.assert_array_equal(apply_window(sig, "rectangular"), sig)


def test_apply_window_reduces_amplitude():
    """Any non-rectangular window tapers the signal; the peak of a windowed constant must be less than 1."""
    sig = np.ones(1024)
    out = apply_window(sig, "hann")
    assert np.max(out) < 1.0


def test_apply_window_unknown_raises():
    """An unrecognised window name must raise ValueError rather than silently applying no window."""
    with pytest.raises(ValueError):
        apply_window(np.ones(64), "unknown_window")


def test_apply_window_hamming_endpoints_near_0_08():
    """The Hamming window has endpoints ≈ 0.08 (not zero), distinguishing it from the Hann window."""
    sig = np.ones(1024)
    out = apply_window(sig, "hamming")
    # np.hamming endpoints are 0.08 exactly
    assert out[0] == pytest.approx(0.08, abs=0.01)
    assert out[-1] == pytest.approx(0.08, abs=0.01)


def test_all_supported_windows_within_unit_range():
    """Every supported window type must produce output within [−1, +1] for a unit-amplitude input."""
    sig = np.ones(1024)
    for window_type in ["hann", "hamming", "blackman", "rectangular"]:
        out = apply_window(sig, window_type)
        assert np.max(out) <= 1.0 + 1e-9
        assert np.min(out) >= -1.0 - 1e-9

