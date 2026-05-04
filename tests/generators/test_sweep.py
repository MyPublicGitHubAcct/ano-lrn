import numpy as np
import pytest

from python.generators import generate_chirp

from tests.generators.conftest import FS, N


def _dominant_frequency(wave: np.ndarray, fs: int) -> float:
    spectrum = np.abs(np.fft.rfft(wave))
    freqs = np.fft.rfftfreq(len(wave), d=1.0 / fs)
    return float(freqs[np.argmax(spectrum)])


def test_chirp_output_shapes():
    t, w = generate_chirp()
    assert t.shape == (N,)
    assert w.shape == (N,)


def test_chirp_amplitude_range(amplitude):
    _, w = generate_chirp(amplitude=amplitude)
    assert np.max(w) <= amplitude + 1e-9
    assert np.min(w) >= -amplitude - 1e-9


def test_chirp_linear_and_log_differ():
    _, w_log = generate_chirp(method="logarithmic")
    _, w_lin = generate_chirp(method="linear")
    assert not np.allclose(w_log, w_lin)


def test_chirp_frequency_increases_over_time(chirp_method):
    _, w = generate_chirp(f_start=100.0, f_end=4000.0, duration=2.0, method=chirp_method)
    seg = len(w) // 10
    assert _dominant_frequency(w[:seg], FS) < _dominant_frequency(w[-seg:], FS)


# --- parameter range tests ---

def test_chirp_downward_sweep(chirp_method):
    _, w = generate_chirp(f_start=4000.0, f_end=100.0, duration=2.0, method=chirp_method)
    seg = len(w) // 10
    assert _dominant_frequency(w[:seg], FS) > _dominant_frequency(w[-seg:], FS)


def test_chirp_amplitude_zero_is_silent():
    _, w = generate_chirp(amplitude=0.0)
    np.testing.assert_array_equal(w, np.zeros(N))
