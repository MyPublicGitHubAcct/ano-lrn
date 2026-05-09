import numpy as np
import pytest

from python.generators import generate_chirp

from tests.generators.conftest import FS, N


def _dominant_frequency(wave: np.ndarray, fs: int) -> float:
    spectrum = np.abs(np.fft.rfft(wave))
    freqs = np.fft.rfftfreq(len(wave), d=1.0 / fs)
    return float(freqs[np.argmax(spectrum)])


def test_chirp_output_shapes():
    """Generator contract: t and signal must share the same N-sample shape."""
    t, w = generate_chirp()
    assert t.shape == (N,)
    assert w.shape == (N,)


def test_chirp_amplitude_range(amplitude):
    """No sample may exceed ±amplitude regardless of instantaneous frequency."""
    _, w = generate_chirp(amplitude=amplitude)
    assert np.max(w) <= amplitude + 1e-9
    assert np.min(w) >= -amplitude - 1e-9


def test_chirp_linear_and_log_differ():
    """Linear and logarithmic sweeps distribute energy differently, so their waveforms must differ."""
    _, w_log = generate_chirp(method="logarithmic")
    _, w_lin = generate_chirp(method="linear")
    assert not np.allclose(w_log, w_lin)


def test_chirp_frequency_increases_over_time(chirp_method):
    """Dominant frequency in the first 10% of the signal must be lower than in the last 10% for an upward sweep."""
    _, w = generate_chirp(f_start=100.0, f_end=4000.0, duration=2.0, method=chirp_method)
    seg = len(w) // 10
    assert _dominant_frequency(w[:seg], FS) < _dominant_frequency(w[-seg:], FS)


# --- parameter range tests ---

def test_chirp_downward_sweep(chirp_method):
    """A downward sweep (f_start > f_end) must show decreasing frequency over time."""
    _, w = generate_chirp(f_start=4000.0, f_end=100.0, duration=2.0, method=chirp_method)
    seg = len(w) // 10
    assert _dominant_frequency(w[:seg], FS) > _dominant_frequency(w[-seg:], FS)


def test_chirp_amplitude_zero_is_silent():
    """amplitude=0 must produce an all-zero signal for both sweep methods."""
    _, w = generate_chirp(amplitude=0.0)
    np.testing.assert_array_equal(w, np.zeros(N))

