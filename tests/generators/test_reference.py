import numpy as np
import pytest

from python.generators import generate_dc, generate_multi_tone

from tests.generators.conftest import FS, N, NYQUIST_PATTERNS


def test_dc_output_shapes():
    """Generator contract: t and signal must share the same N-sample shape."""
    t, w = generate_dc()
    assert t.shape == (N,)
    assert w.shape == (N,)


def test_dc_constant_value(dc_amplitude):
    """Every sample must equal the requested amplitude; any variation means the generator is not truly DC."""
    _, w = generate_dc(amplitude=dc_amplitude)
    assert np.all(w == pytest.approx(dc_amplitude))


def test_nyquist_output_shapes(nyquist_gen):
    """Generator contract: t and signal must share the same N-sample shape."""
    t, w = nyquist_gen()
    assert t.shape == (N,)
    assert w.shape == (N,)


def test_nyquist_sample_pattern(nyquist_gen):
    """The first samples must match the analytic pattern for each Nyquist reference signal."""
    _, w = nyquist_gen(fs=8, duration=1.0)
    expected = NYQUIST_PATTERNS[nyquist_gen]
    np.testing.assert_allclose(w[:len(expected)], expected, atol=1e-9)


def test_nyquist_amplitude_scaling(nyquist_gen, amplitude):
    """Peak absolute value must equal the requested amplitude for all Nyquist generators."""
    _, w = nyquist_gen(amplitude=amplitude)
    assert np.max(np.abs(w)) == pytest.approx(amplitude, abs=1e-9)


def test_multi_tone_output_shape(multi_tone_freqs):
    """Generator contract: t and signal must share the same N-sample shape."""
    t, w = generate_multi_tone(freqs=multi_tone_freqs)
    assert t.shape == (N,)
    assert w.shape == (N,)


def test_multi_tone_peak_amplitude(multi_tone_freqs, amplitude):
    """Normalized multi-tone must reach exactly amplitude at its loudest point."""
    _, w = generate_multi_tone(freqs=multi_tone_freqs, amplitude=amplitude)
    assert np.max(np.abs(w)) == pytest.approx(amplitude, abs=1e-9)


def test_multi_tone_contains_expected_frequencies():
    """Each requested frequency must appear as a prominent spectral peak (> 30% of the maximum)."""
    freqs = [220.0, 880.0, 3520.0]
    _, w = generate_multi_tone(freqs=freqs)
    spectrum = np.abs(np.fft.rfft(w))
    freq_axis = np.fft.rfftfreq(len(w), d=1.0 / FS)
    peak = np.max(spectrum)
    for f in freqs:
        idx = int(np.argmin(np.abs(freq_axis - f)))
        assert spectrum[idx] > 0.3 * peak


def test_multi_tone_single_freq_matches_sine():
    """generate_multi_tone with one frequency must be proportional to generate_sine (same waveform shape)."""
    from python.generators import generate_sine
    freq = 440.0
    _, w_multi = generate_multi_tone(freqs=[freq])
    _, w_sine = generate_sine(freq=freq)
    # multi_tone normalises by its own peak; verify same waveform up to that constant scale
    peak = np.max(np.abs(w_sine)) + 1e-12
    np.testing.assert_allclose(w_multi, w_sine / peak, atol=1e-9)


def test_dc_amplitude_zero_is_silent():
    """generate_dc with amplitude=0 must return an all-zero signal."""
    _, w = generate_dc(amplitude=0.0)
    np.testing.assert_array_equal(w, np.zeros(N))

