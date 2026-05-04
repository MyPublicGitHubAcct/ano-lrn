import numpy as np
import pytest

from python.generators import generate_dc, generate_multi_tone

from tests.generators.conftest import FS, N, NYQUIST_PATTERNS


def test_dc_output_shapes():
    t, w = generate_dc()
    assert t.shape == (N,)
    assert w.shape == (N,)


def test_dc_constant_value(dc_amplitude):
    _, w = generate_dc(amplitude=dc_amplitude)
    assert np.all(w == pytest.approx(dc_amplitude))


def test_nyquist_output_shapes(nyquist_gen):
    t, w = nyquist_gen()
    assert t.shape == (N,)
    assert w.shape == (N,)


def test_nyquist_sample_pattern(nyquist_gen):
    _, w = nyquist_gen(fs=8, duration=1.0)
    expected = NYQUIST_PATTERNS[nyquist_gen]
    np.testing.assert_allclose(w[:len(expected)], expected, atol=1e-9)


def test_nyquist_amplitude_scaling(nyquist_gen, amplitude):
    _, w = nyquist_gen(amplitude=amplitude)
    assert np.max(np.abs(w)) == pytest.approx(amplitude, abs=1e-9)


def test_multi_tone_output_shape(multi_tone_freqs):
    t, w = generate_multi_tone(freqs=multi_tone_freqs)
    assert t.shape == (N,)
    assert w.shape == (N,)


def test_multi_tone_peak_amplitude(multi_tone_freqs, amplitude):
    _, w = generate_multi_tone(freqs=multi_tone_freqs, amplitude=amplitude)
    assert np.max(np.abs(w)) == pytest.approx(amplitude, abs=1e-9)


def test_multi_tone_contains_expected_frequencies():
    freqs = [220.0, 880.0, 3520.0]
    _, w = generate_multi_tone(freqs=freqs)
    spectrum = np.abs(np.fft.rfft(w))
    freq_axis = np.fft.rfftfreq(len(w), d=1.0 / FS)
    peak = np.max(spectrum)
    for f in freqs:
        idx = int(np.argmin(np.abs(freq_axis - f)))
        assert spectrum[idx] > 0.3 * peak
