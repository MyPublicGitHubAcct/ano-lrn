import numpy as np

from python.source_filter import formant_filter
from python.generators import generate_impulse

FS = 44100


def _impulse(n=1024):
    _, imp = generate_impulse(fs=FS, duration=n / FS)
    return imp


def test_formant_filter_shape():
    sig = _impulse()
    out = formant_filter(sig, formant_freqs=[500.0, 1500.0], bandwidths=[80.0, 120.0], fs=FS)
    assert out.shape == sig.shape


def test_formant_filter_boosts_formant_frequency():
    sig = _impulse(2048)
    out = formant_filter(sig, formant_freqs=[1000.0], bandwidths=[100.0], fs=FS)
    spectrum = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(len(out), d=1.0 / FS)
    mask_1k = (freqs >= 900) & (freqs <= 1100)
    mask_dc = freqs < 100
    energy_1k = np.sum(spectrum[mask_1k] ** 2)
    energy_dc = np.sum(spectrum[mask_dc] ** 2)
    assert energy_1k > energy_dc
