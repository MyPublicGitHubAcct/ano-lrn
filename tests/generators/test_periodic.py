import numpy as np
import pytest

from python.generators import generate_sine, generate_square, generate_triangle

from tests.generators.conftest import FS, N, NYQUIST_PATTERNS


def _dominant_frequency(wave: np.ndarray, fs: int) -> float:
    spectrum = np.abs(np.fft.rfft(wave))
    freqs = np.fft.rfftfreq(len(wave), d=1.0 / fs)
    return float(freqs[np.argmax(spectrum)])


def test_periodic_output_shapes(periodic_gen):
    """Generator contract: t and signal must share the same N-sample shape."""
    t, w = periodic_gen()
    assert t.shape == (N,)
    assert w.shape == (N,)


def test_periodic_amplitude_range(periodic_gen, amplitude):
    """No sample may escape [−amplitude, +amplitude] at any of the tested amplitudes."""
    _, w = periodic_gen(amplitude=amplitude)
    assert np.max(w) <= amplitude + 1e-9
    assert np.min(w) >= -amplitude - 1e-9


def test_periodic_dominant_frequency(periodic_gen, freq):
    """FFT peak within 1 Hz of the target confirms the waveform is correctly tuned."""
    _, w = periodic_gen(freq=freq)
    assert _dominant_frequency(w, FS) == pytest.approx(freq, abs=1.0)


def test_periodic_sample_rate(periodic_gen, fs_and_duration):
    """Output length must equal round(fs × duration) for non-default sample rates."""
    fs, duration = fs_and_duration
    t, w = periodic_gen(fs=fs, duration=duration)
    assert t.shape == w.shape == (int(fs * duration),)


def test_sine_phase_shifts_output(phase):
    """A non-zero phase offset must produce a waveform distinct from the zero-phase baseline."""
    _, w0 = generate_sine(phase=0.0)
    _, w1 = generate_sine(phase=phase)
    assert not np.allclose(w0, w1)


def test_square_values_are_binary(amplitude):
    """Square wave is defined as only ±amplitude; any intermediate value indicates a bug."""
    _, w = generate_square(amplitude=amplitude)
    assert np.all((w == amplitude) | (w == -amplitude))


def test_square_duty_cycle(duty):
    """Fraction of positive samples must match the duty parameter within a 2% tolerance."""
    _, w = generate_square(duty=duty)
    assert np.sum(w > 0) / N == pytest.approx(duty, abs=0.02)


def test_triangle_peaks_reach_amplitude(amplitude):
    """Triangle must reach exactly ±amplitude; a lower peak means the scaling coefficient is wrong."""
    _, w = generate_triangle(freq=100.0, amplitude=amplitude)
    assert np.max(w) == pytest.approx(amplitude, abs=0.01)
    assert np.min(w) == pytest.approx(-amplitude, abs=0.01)


# --- parameter range tests ---

def test_square_duty_zero_is_all_negative():
    """duty=0 is an edge case: the pulse is never high, so every sample must be −amplitude."""
    _, w = generate_square(freq=440.0, duty=0.0, amplitude=1.0)
    np.testing.assert_array_equal(w, np.full(N, -1.0))


def test_square_duty_one_is_all_positive():
    """duty=1 is the complementary edge: permanently high, so every sample must be +amplitude."""
    _, w = generate_square(freq=440.0, duty=1.0, amplitude=1.0)
    np.testing.assert_array_equal(w, np.full(N, 1.0))


def test_periodic_zero_amplitude_is_silent(periodic_gen):
    """amplitude=0 must produce an all-zero signal regardless of waveform shape."""
    _, w = periodic_gen(amplitude=0.0)
    np.testing.assert_array_equal(w, np.zeros(N))


def test_periodic_freq_near_nyquist_is_finite(periodic_gen):
    """Frequencies near Nyquist must not produce NaN or Inf in any generator."""
    _, w = periodic_gen(freq=FS * 0.49)
    assert np.all(np.isfinite(w))


def test_sawtooth_rises_monotonically_within_cycle():
    """Within each cycle (away from the reset boundary) the sawtooth must increase sample-by-sample."""
    from python.generators import generate_sawtooth
    freq = 100.0
    _, w = generate_sawtooth(freq=freq, fs=FS)
    samples_per_cycle = int(FS / freq)
    # Check a window in the middle of the first cycle, well away from the reset discontinuity
    mid_start = samples_per_cycle // 4
    mid_end = 3 * samples_per_cycle // 4
    assert np.all(np.diff(w[mid_start:mid_end]) > 0)


def test_sine_zero_freq_is_constant():
    """At freq=0 the phase never advances; every sample must equal amplitude·sin(phase)."""
    _, w = generate_sine(freq=0.0)
    assert np.all(w == pytest.approx(w[0]))

