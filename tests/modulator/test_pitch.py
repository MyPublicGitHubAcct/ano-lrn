import numpy as np

from python.modulator import vibrato

FS = 44100


def _sine(freq=440.0, n=4096):
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * freq * t)


def test_vibrato_shape():
    """vibrato must return an array of the same length as the input."""
    sig = _sine(n=512)
    assert vibrato(sig, rate=5.0, depth_samples=3.0, fs=FS).shape == sig.shape


def test_vibrato_output_bounded():
    """With a unit-amplitude sine input, vibrato must not produce samples outside roughly ±1 (linear interpolation boundary)."""
    sig = _sine(n=1024)
    out = vibrato(sig, rate=5.0, depth_samples=3.0, fs=FS)
    assert np.max(np.abs(out)) <= 1.05


def test_vibrato_depth_zero_produces_constant_delay():
    """With depth_samples=0, the delay is a fixed 1-sample offset; output must be finite and same length as input."""
    sig = _sine(n=1024)
    out = vibrato(sig, rate=5.0, depth_samples=0, fs=FS)
    assert out.shape == sig.shape
    assert np.all(np.isfinite(out))


def test_vibrato_large_depth_no_crash():
    """Very large depth_samples must not cause IndexError or NaN (should remain bounded)."""
    n = 4096
    sig = _sine(n=n)
    out = vibrato(sig, rate=1.0, depth_samples=n // 4, fs=FS)
    assert out.shape == sig.shape
    assert np.all(np.isfinite(out))

