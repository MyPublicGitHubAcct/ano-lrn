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

