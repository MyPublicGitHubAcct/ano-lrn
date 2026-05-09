import numpy as np

from python.source_filter import formant_filter
from python.generators import generate_impulse

FS = 44100


def _impulse(n=1024):
    _, imp = generate_impulse(fs=FS, duration=n / FS)
    return imp


def test_formant_filter_shape():
    """formant_filter must return an array of the same length as the input."""
    sig = _impulse()
    out = formant_filter(sig, formant_freqs=[500.0, 1500.0], bandwidths=[80.0, 120.0], fs=FS)
    assert out.shape == sig.shape


def test_formant_filter_boosts_formant_frequency():
    """The spectral region around the formant frequency must have more energy than DC after impuse excitation."""
    sig = _impulse(2048)
    out = formant_filter(sig, formant_freqs=[1000.0], bandwidths=[100.0], fs=FS)
    spectrum = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(len(out), d=1.0 / FS)
    mask_1k = (freqs >= 900) & (freqs <= 1100)
    mask_dc = freqs < 100
    energy_1k = np.sum(spectrum[mask_1k] ** 2)
    energy_dc = np.sum(spectrum[mask_dc] ** 2)
    assert energy_1k > energy_dc


def test_formant_filter_two_formants_produce_two_peaks():
    """Two formants must each produce a spectral peak; both formant frequencies must appear in the top spectral peaks."""
    sig = _impulse(8192)
    f1, f2 = 500.0, 1500.0
    out = formant_filter(sig, formant_freqs=[f1, f2], bandwidths=[80.0, 120.0], fs=FS)
    spectrum = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(len(out), d=1.0 / FS)
    # Energy at each formant frequency must exceed the spectral floor (mean over the full range)
    floor = np.mean(spectrum)
    for f in [f1, f2]:
        idx = np.argmin(np.abs(freqs - f))
        assert spectrum[idx] > floor * 2


def test_formant_filter_narrower_bandwidth_sharper_peak():
    """Narrower bandwidth must concentrate more energy in a ±50 Hz window around the formant."""
    sig = _impulse(4096)
    f0 = 1000.0
    out_wide = formant_filter(sig, formant_freqs=[f0], bandwidths=[400.0], fs=FS)
    out_narrow = formant_filter(sig, formant_freqs=[f0], bandwidths=[50.0], fs=FS)
    spec_wide = np.abs(np.fft.rfft(out_wide))
    spec_narrow = np.abs(np.fft.rfft(out_narrow))
    freqs = np.fft.rfftfreq(len(out_wide), d=1.0 / FS)
    mask = (freqs >= f0 - 50) & (freqs <= f0 + 50)
    total_wide = np.sum(spec_wide ** 2)
    total_narrow = np.sum(spec_narrow ** 2)
    frac_wide = np.sum(spec_wide[mask] ** 2) / (total_wide + 1e-12)
    frac_narrow = np.sum(spec_narrow[mask] ** 2) / (total_narrow + 1e-12)
    assert frac_narrow > frac_wide


def test_formant_filter_empty_freqs_returns_input_unchanged():
    """formant_filter with an empty formant_freqs list must return the input signal unmodified."""
    sig = _impulse(1024)
    out = formant_filter(sig, formant_freqs=[], bandwidths=[], fs=FS)
    np.testing.assert_array_equal(out, sig)

