import numpy as np
import pytest

from python.spectral import spectral_centroid, spectral_flux, spectral_gate

FS = 44100
FRAME = 512
HOP = 128


def _sine(freq=440.0, n=4096):
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * freq * t)


def _rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


# ── Shape ─────────────────────────────────────────────────────────────────────

def test_spectral_centroid_shape():
    sig = _sine(n=4096)
    c = spectral_centroid(sig, FS, FRAME, HOP)
    assert c.ndim == 1
    assert len(c) > 0


def test_spectral_flux_shape():
    sig = _sine(n=4096)
    f = spectral_flux(sig, FRAME, HOP)
    assert f.ndim == 1
    assert len(f) > 0


def test_spectral_gate_shape():
    sig = _sine(n=4096)
    out = spectral_gate(sig, threshold_db=-40.0, frame_size=FRAME, hop_size=HOP)
    assert out.shape == sig.shape


# ── Spectral centroid ─────────────────────────────────────────────────────────

def test_spectral_centroid_high_freq_higher_than_low_freq():
    sig_low = _sine(freq=200.0, n=4096)
    sig_high = _sine(freq=4000.0, n=4096)
    c_low = np.mean(spectral_centroid(sig_low, FS, FRAME, HOP))
    c_high = np.mean(spectral_centroid(sig_high, FS, FRAME, HOP))
    assert c_high > c_low


def test_spectral_centroid_within_audio_band():
    sig = _sine(freq=1000.0, n=4096)
    c = spectral_centroid(sig, FS, FRAME, HOP)
    assert np.all(c >= 0.0)
    assert np.all(c <= FS / 2)


# ── Spectral flux ─────────────────────────────────────────────────────────────

def test_spectral_flux_first_frame_is_zero():
    sig = _sine(n=4096)
    f = spectral_flux(sig, FRAME, HOP)
    assert f[0] == pytest.approx(0.0)


def test_spectral_flux_steady_sine_low():
    # A stationary sine has nearly constant spectrum → low flux.
    sig = _sine(freq=440.0, n=4096)
    f = spectral_flux(sig, FRAME, HOP)
    # Ignore first frame (onset) and check that steady-state flux is small.
    assert np.mean(f[2:]) < 100.0


def test_spectral_flux_noise_higher_than_sine():
    n = 4096
    rng = np.random.default_rng(0)
    sig_sine = _sine(n=n)
    sig_noise = rng.standard_normal(n)
    f_sine = np.mean(spectral_flux(sig_sine, FRAME, HOP)[1:])
    f_noise = np.mean(spectral_flux(sig_noise, FRAME, HOP)[1:])
    assert f_noise > f_sine


# ── Spectral gate ─────────────────────────────────────────────────────────────

def test_spectral_gate_loud_signal_passes():
    sig = _sine(n=4096)
    out = spectral_gate(sig, threshold_db=-60.0, frame_size=FRAME, hop_size=HOP)
    # Most of the sine should survive a very low threshold.
    assert _rms(out) > 0.1


def test_spectral_gate_silence_below_threshold():
    # Silent signal (all bins below threshold) → output near zero.
    sig = np.zeros(4096)
    out = spectral_gate(sig, threshold_db=-10.0, frame_size=FRAME, hop_size=HOP)
    np.testing.assert_allclose(out, np.zeros_like(out), atol=1e-6)


def test_spectral_gate_high_threshold_attenuates():
    sig = _sine(n=4096)
    # Raw STFT bins for a windowed frame scale with frame size (peak ≈ 42 dB at 512).
    # A threshold well above the peak silences the output.
    out = spectral_gate(sig, threshold_db=80.0, frame_size=FRAME, hop_size=HOP)
    assert _rms(out) < _rms(sig) * 0.01
