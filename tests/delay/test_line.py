import numpy as np
import pytest

from python.delay import delay_line, fractional_delay_line
from python.generators import generate_impulse

FS = 44100


def _impulse(n=1024):
    _, imp = generate_impulse(fs=FS, duration=n / FS)
    return imp


def _sine(freq=440.0, n=1024):
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * freq * t)


def test_delay_line_shape():
    """delay_line must return an array of the same length as the input."""
    assert delay_line(_sine(), 100).shape == _sine().shape


def test_delay_line_zero_delay_is_identity():
    """A zero-sample delay must return the signal unmodified — the trivial pass-through case."""
    sig = _sine()
    np.testing.assert_array_equal(delay_line(sig, 0), sig)


def test_delay_line_shifts_samples():
    """An impulse at index 0 must appear at index d after a d-sample delay, confirming the offset is exact."""
    sig = _impulse()
    d = 50
    out = delay_line(sig, d)
    assert out[d] == pytest.approx(1.0)
    assert np.all(out[:d] == 0.0)


def test_delay_line_fills_zeros_at_start():
    """The first d samples of the output must be zero (no wrap-around from the end of the signal)."""
    sig = np.ones(100)
    out = delay_line(sig, 10)
    np.testing.assert_array_equal(out[:10], np.zeros(10))
    np.testing.assert_array_equal(out[10:], np.ones(90))


def test_delay_line_delay_exceeds_length_is_all_zero():
    """When delay >= len(signal), the signal is pushed entirely beyond the buffer; output must be all-zero."""
    sig = _sine()
    out = delay_line(sig, len(sig))
    np.testing.assert_array_equal(out, np.zeros_like(sig))


def test_delay_line_fractional_delay_is_truncated():
    """Only integer delays are supported; a fractional value must be truncated, not rounded."""
    sig = _impulse()
    d_float = 50.7
    d_int = int(d_float)  # truncates to 50
    out = delay_line(sig, d_float)
    assert out[d_int] == pytest.approx(1.0)
    assert np.all(out[:d_int] == 0.0)


# ---------------------------------------------------------------------------
# fractional_delay_line
# ---------------------------------------------------------------------------


def test_fractional_delay_shape():
    """fractional_delay_line must return an array of the same length as the input."""
    sig = _sine()
    assert fractional_delay_line(sig, 7.3).shape == sig.shape


def test_fractional_delay_zero_is_identity():
    """A delay of 0.0 samples must return the input signal unchanged."""
    sig = _sine()
    np.testing.assert_allclose(fractional_delay_line(sig, 0.0), sig, atol=1e-12)


def test_fractional_delay_integer_matches_delay_line():
    """When delay_samples is an exact integer float, output must equal delay_line's output."""
    sig = _sine()
    d = 50
    np.testing.assert_allclose(
        fractional_delay_line(sig, float(d)), delay_line(sig, d), atol=1e-12
    )


def test_fractional_delay_impulse_center_of_mass():
    """The center of mass of the impulse response must equal the requested delay.

    For an impulse input, Lagrange interpolation distributes energy across a
    small window; the weighted centroid of those samples must land at exactly
    delay_samples, proving the fractional offset is correct.
    """
    sig = _impulse(n=256)
    d = 10.7
    out = fractional_delay_line(sig, d)
    n = np.arange(len(out))
    center_of_mass = np.sum(n * out) / np.sum(out)
    assert center_of_mass == pytest.approx(d, abs=1e-10)


def test_fractional_delay_half_sample_symmetric():
    """A 0.5-sample fractional delay must produce a symmetric impulse response.

    At exactly half a sample the two central Lagrange weights are equal, so the
    energy on each side of the fractional position must be identical.
    """
    sig = _impulse(n=128)
    d_int = 5
    out = fractional_delay_line(sig, d_int + 0.5)
    # The two central taps at d_int and d_int+1 must be equal
    assert out[d_int] == pytest.approx(out[d_int + 1], rel=1e-10)


def test_fractional_delay_sine_amplitude_preserved():
    """A pure tone passed through fractional_delay_line must retain its amplitude.

    A delay (fractional or integer) is an all-pass operation; the FFT magnitude
    at the dominant frequency must match the input's to within 1%.
    """
    sig = _sine(freq=440.0, n=4096)
    out = fractional_delay_line(sig, 13.6)
    peak_in = np.max(np.abs(np.fft.rfft(sig)))
    peak_out = np.max(np.abs(np.fft.rfft(out)))
    assert peak_out == pytest.approx(peak_in, rel=0.01)


def test_fractional_delay_order4_matches_identity_at_integer():
    """Order-4 Lagrange must also return delay_line output for integer delays."""
    sig = _sine()
    d = 30
    np.testing.assert_allclose(
        fractional_delay_line(sig, float(d), order=4), delay_line(sig, d), atol=1e-12
    )


def test_fractional_delay_order4_center_of_mass():
    """Order-4 interpolation must also place the impulse centroid at delay_samples."""
    sig = _impulse(n=256)
    d = 8.3
    out = fractional_delay_line(sig, d, order=4)
    n = np.arange(len(out))
    center_of_mass = np.sum(n * out) / np.sum(out)
    assert center_of_mass == pytest.approx(d, abs=1e-10)


def test_fractional_delay_varying_delay_shape():
    """Passing an array of per-sample delays must return an array of the same length."""
    sig = _sine(n=512)
    delay_arr = np.linspace(5.0, 15.0, len(sig))
    out = fractional_delay_line(sig, delay_arr)
    assert out.shape == sig.shape


def test_fractional_delay_varying_delay_finite():
    """Per-sample delay array must not produce NaN or Inf."""
    sig = _sine(n=512)
    delay_arr = 10.0 + 4.0 * np.sin(2 * np.pi * np.arange(len(sig)) / len(sig))
    out = fractional_delay_line(sig, delay_arr)
    assert np.all(np.isfinite(out))


def test_fractional_delay_invalid_order():
    """order values other than 3 or 4 must raise ValueError."""
    sig = _sine()
    with pytest.raises(ValueError):
        fractional_delay_line(sig, 5.5, order=2)

