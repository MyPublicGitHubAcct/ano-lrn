import numpy as np
import pytest

from python.modulator import ring_modulate, tremolo

FS = 44100


def _sine(freq=440.0, n=4096):
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * freq * t)


def test_tremolo_shape():
    """tremolo must return an array of the same length as the input."""
    sig = _sine()
    assert tremolo(sig, rate=5.0, depth=0.5, fs=FS).shape == sig.shape


def test_ring_modulate_shape():
    """ring_modulate must return an array of the same length as the input."""
    sig = _sine()
    assert ring_modulate(sig, carrier_freq=200.0, fs=FS).shape == sig.shape


def test_tremolo_zero_depth_is_identity():
    """depth=0 means the LFO contributes nothing; the output must equal the unmodulated input."""
    sig = _sine()
    np.testing.assert_allclose(tremolo(sig, rate=5.0, depth=0.0, fs=FS), sig, atol=1e-12)


def test_tremolo_full_depth_amplitude_range():
    """depth=1.0 causes the envelope to swing fully between 0 and 1; these extremes must be reached."""
    n = FS
    sig = np.ones(n)
    out = tremolo(sig, rate=2.0, depth=1.0, fs=FS)
    assert np.min(out) == pytest.approx(0.0, abs=1e-6)
    assert np.max(out) == pytest.approx(1.0, abs=1e-6)


def test_tremolo_does_not_change_sign():
    """Tremolo multiplies by a positive envelope; a positive-constant input must remain non-negative."""
    sig = np.ones(FS)
    out = tremolo(sig, rate=3.0, depth=0.9, fs=FS)
    assert np.all(out >= 0.0)


def test_ring_modulate_produces_sidebands():
    """Ring modulation of a 1 kHz signal by a 200 Hz carrier must produce sidebands at 800 Hz and 1200 Hz."""
    n = FS
    sig = np.sin(2 * np.pi * 1000.0 * np.arange(n) / FS)
    out = ring_modulate(sig, carrier_freq=200.0, fs=FS)
    spectrum = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(n, d=1.0 / FS)
    peak_freqs = freqs[np.argsort(spectrum)[-4:]]
    assert any(np.abs(peak_freqs - 800.0) < 5.0)
    assert any(np.abs(peak_freqs - 1200.0) < 5.0)


def test_ring_modulate_by_dc_is_identity():
    """A 0 Hz carrier is a constant 1.0; the ring modulated output must equal the input."""
    sig = _sine()
    out = ring_modulate(sig, carrier_freq=0.0, fs=FS)
    np.testing.assert_allclose(out, sig, atol=1e-12)


def test_tremolo_lfo_period_is_one_second_at_rate_one():
    """At rate=1 Hz with depth=1, the tremolo envelope must complete exactly one full cycle per second."""
    n = FS  # exactly 1 second
    sig = np.ones(n)
    out = tremolo(sig, rate=1.0, depth=1.0, fs=FS)
    # The LFO is (1 - 0.5*(1 - cos(2π*t))); in 1 second it returns to its starting value
    assert out[0] == pytest.approx(out[-1], abs=0.01)
    # And there should be exactly one minimum (trough) in the output
    troughs = np.where((out[1:-1] < out[:-2]) & (out[1:-1] < out[2:]))[0]
    assert len(troughs) == 1


def test_ring_modulate_signal_freq_carrier_produces_double_frequency():
    """ring_modulate by cos(2πft) applied to sin(2πft) gives sin*cos = 0.5·sin(4πft); dominant freq is 2×."""
    n = FS
    freq = 440.0
    sig = np.sin(2 * np.pi * freq * np.arange(n) / FS)
    out = ring_modulate(sig, carrier_freq=freq, fs=FS)
    spectrum = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(n, d=1.0 / FS)
    peak_freq = freqs[np.argmax(spectrum)]
    assert abs(peak_freq - 2 * freq) < 5.0

