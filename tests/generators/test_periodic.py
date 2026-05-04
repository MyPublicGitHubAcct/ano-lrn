import numpy as np
import pytest

from python.generators import generate_sine, generate_square, generate_triangle

from tests.generators.conftest import FS, N, NYQUIST_PATTERNS


def _dominant_frequency(wave: np.ndarray, fs: int) -> float:
    spectrum = np.abs(np.fft.rfft(wave))
    freqs = np.fft.rfftfreq(len(wave), d=1.0 / fs)
    return float(freqs[np.argmax(spectrum)])


def test_periodic_output_shapes(periodic_gen):
    t, w = periodic_gen()
    assert t.shape == (N,)
    assert w.shape == (N,)


def test_periodic_amplitude_range(periodic_gen, amplitude):
    _, w = periodic_gen(amplitude=amplitude)
    assert np.max(w) <= amplitude + 1e-9
    assert np.min(w) >= -amplitude - 1e-9


def test_periodic_dominant_frequency(periodic_gen, freq):
    _, w = periodic_gen(freq=freq)
    assert _dominant_frequency(w, FS) == pytest.approx(freq, abs=1.0)


def test_periodic_sample_rate(periodic_gen, fs_and_duration):
    fs, duration = fs_and_duration
    t, w = periodic_gen(fs=fs, duration=duration)
    assert t.shape == w.shape == (int(fs * duration),)


def test_sine_phase_shifts_output(phase):
    _, w0 = generate_sine(phase=0.0)
    _, w1 = generate_sine(phase=phase)
    assert not np.allclose(w0, w1)


def test_square_values_are_binary(amplitude):
    _, w = generate_square(amplitude=amplitude)
    assert np.all((w == amplitude) | (w == -amplitude))


def test_square_duty_cycle(duty):
    _, w = generate_square(duty=duty)
    assert np.sum(w > 0) / N == pytest.approx(duty, abs=0.02)


def test_triangle_peaks_reach_amplitude(amplitude):
    _, w = generate_triangle(freq=100.0, amplitude=amplitude)
    assert np.max(w) == pytest.approx(amplitude, abs=0.01)
    assert np.min(w) == pytest.approx(-amplitude, abs=0.01)


# --- parameter range tests ---

def test_square_duty_zero_is_all_negative():
    _, w = generate_square(freq=440.0, duty=0.0, amplitude=1.0)
    np.testing.assert_array_equal(w, np.full(N, -1.0))


def test_square_duty_one_is_all_positive():
    _, w = generate_square(freq=440.0, duty=1.0, amplitude=1.0)
    np.testing.assert_array_equal(w, np.full(N, 1.0))


def test_periodic_zero_amplitude_is_silent(periodic_gen):
    _, w = periodic_gen(amplitude=0.0)
    np.testing.assert_array_equal(w, np.zeros(N))


def test_periodic_freq_near_nyquist_is_finite(periodic_gen):
    _, w = periodic_gen(freq=FS * 0.49)
    assert np.all(np.isfinite(w))
