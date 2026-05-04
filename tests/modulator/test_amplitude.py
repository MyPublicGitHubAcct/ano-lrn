import numpy as np
import pytest

from python.modulator import ring_modulate, tremolo

FS = 44100


def _sine(freq=440.0, n=4096):
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * freq * t)


def test_tremolo_shape():
    sig = _sine()
    assert tremolo(sig, rate=5.0, depth=0.5, fs=FS).shape == sig.shape


def test_ring_modulate_shape():
    sig = _sine()
    assert ring_modulate(sig, carrier_freq=200.0, fs=FS).shape == sig.shape


def test_tremolo_zero_depth_is_identity():
    sig = _sine()
    np.testing.assert_allclose(tremolo(sig, rate=5.0, depth=0.0, fs=FS), sig, atol=1e-12)


def test_tremolo_full_depth_amplitude_range():
    n = FS
    sig = np.ones(n)
    out = tremolo(sig, rate=2.0, depth=1.0, fs=FS)
    assert np.min(out) == pytest.approx(0.0, abs=1e-6)
    assert np.max(out) == pytest.approx(1.0, abs=1e-6)


def test_tremolo_does_not_change_sign():
    sig = np.ones(FS)
    out = tremolo(sig, rate=3.0, depth=0.9, fs=FS)
    assert np.all(out >= 0.0)


def test_ring_modulate_produces_sidebands():
    n = FS
    sig = np.sin(2 * np.pi * 1000.0 * np.arange(n) / FS)
    out = ring_modulate(sig, carrier_freq=200.0, fs=FS)
    spectrum = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(n, d=1.0 / FS)
    peak_freqs = freqs[np.argsort(spectrum)[-4:]]
    assert any(np.abs(peak_freqs - 800.0) < 5.0)
    assert any(np.abs(peak_freqs - 1200.0) < 5.0)


def test_ring_modulate_by_dc_is_identity():
    sig = _sine()
    out = ring_modulate(sig, carrier_freq=0.0, fs=FS)
    np.testing.assert_allclose(out, sig, atol=1e-12)
