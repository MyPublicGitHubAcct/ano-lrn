import numpy as np

from python.spectral import spectral_gate

FS = 44100
FRAME = 512
HOP = 128


def _sine(freq=440.0, n=4096):
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * freq * t)


def _rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


def test_spectral_gate_shape():
    sig = _sine(n=4096)
    out = spectral_gate(sig, threshold_db=-40.0, frame_size=FRAME, hop_size=HOP)
    assert out.shape == sig.shape


def test_spectral_gate_loud_signal_passes():
    sig = _sine(n=4096)
    out = spectral_gate(sig, threshold_db=-60.0, frame_size=FRAME, hop_size=HOP)
    assert _rms(out) > 0.1


def test_spectral_gate_silence_below_threshold():
    sig = np.zeros(4096)
    out = spectral_gate(sig, threshold_db=-10.0, frame_size=FRAME, hop_size=HOP)
    np.testing.assert_allclose(out, np.zeros_like(out), atol=1e-6)


def test_spectral_gate_high_threshold_attenuates():
    sig = _sine(n=4096)
    out = spectral_gate(sig, threshold_db=80.0, frame_size=FRAME, hop_size=HOP)
    assert _rms(out) < _rms(sig) * 0.01
