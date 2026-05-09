import numpy as np
import pytest

from python.source_separation import wiener_filter

FS = 44100
FRAME = 512
HOP = 128


def _sine(freq=440.0, n=4096):
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * freq * t)


def _rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


def test_wiener_filter_output_shape():
    """wiener_filter must return an array with the same length as the mixture."""
    mixture = _sine(440.0, n=4096)
    estimate = _sine(440.0, n=4096) * 0.9
    out = wiener_filter(mixture, estimate, frame_size=FRAME, hop_size=HOP)
    assert out.shape == mixture.shape


def test_wiener_filter_identical_estimate_passes_signal():
    """When the estimate perfectly matches the mixture, the Wiener mask is 1 everywhere and the signal passes."""
    sig = _sine(n=4096)
    out = wiener_filter(sig, sig, frame_size=FRAME, hop_size=HOP)
    assert _rms(out) == pytest.approx(_rms(sig), rel=0.1)


def test_wiener_filter_zero_estimate_suppresses_signal():
    """A zero estimate gives a Wiener mask of 0 everywhere; the output must be essentially silent."""
    sig = _sine(n=4096)
    estimate = np.zeros_like(sig)
    out = wiener_filter(sig, estimate, frame_size=FRAME, hop_size=HOP)
    assert _rms(out) < _rms(sig) * 0.01


def test_wiener_filter_partial_estimate_monotone_suppression():
    """A half-level estimate (mixture × 0.5) must yield output RMS between 0 and the full-pass level."""
    sig = _sine(n=4096)
    estimate_full = sig.copy()
    estimate_half = sig * 0.5
    out_full = wiener_filter(sig, estimate_full, frame_size=FRAME, hop_size=HOP)
    out_half = wiener_filter(sig, estimate_half, frame_size=FRAME, hop_size=HOP)
    out_zero = wiener_filter(sig, np.zeros_like(sig), frame_size=FRAME, hop_size=HOP)
    assert _rms(out_zero) < _rms(out_half) < _rms(out_full)


def test_wiener_filter_frequency_selective_estimate():
    """An estimate covering only low frequencies must pass low-frequency content and suppress high-frequency."""
    n = 4096
    t = np.arange(n) / FS
    low_tone = np.sin(2 * np.pi * 200.0 * t)
    high_tone = np.sin(2 * np.pi * 4000.0 * t)
    mixture = low_tone + high_tone
    # Estimate = low tone only
    out = wiener_filter(mixture, low_tone, frame_size=FRAME, hop_size=HOP)
    spectrum = np.abs(np.fft.rfft(out))
    freqs_ax = np.fft.rfftfreq(n, d=1.0 / FS)
    energy_low = np.sum(spectrum[(freqs_ax >= 100) & (freqs_ax <= 500)] ** 2)
    energy_high = np.sum(spectrum[(freqs_ax >= 2000) & (freqs_ax <= 6000)] ** 2)
    assert energy_low > energy_high * 2

