import numpy as np
import pytest

from python.delay import pitch_shift

FS = 44100


def _sine(freq=440.0, duration=2.0):
    N = int(duration * FS)
    t = np.arange(N) / FS
    return np.sin(2 * np.pi * freq * t)


def _buf_size(fs=FS):
    B = 1
    while B < int(fs * 0.05):
        B <<= 1
    return B


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

def test_output_length():
    sig = _sine()
    out = pitch_shift(sig, FS, semitones=7)
    assert len(out) == len(sig)


def test_output_length_odd_input():
    sig = np.random.default_rng(0).standard_normal(33333)
    out = pitch_shift(sig, FS, semitones=-3)
    assert len(out) == len(sig)


# ---------------------------------------------------------------------------
# semitones=0 passthrough
# ---------------------------------------------------------------------------

def test_semitones_zero_passthrough():
    """out[B//2:] must equal signal[:N-B//2] to floating-point precision."""
    sig = _sine(duration=1.0)
    out = pitch_shift(sig, FS, semitones=0)
    B = _buf_size()
    N = len(sig)
    np.testing.assert_allclose(out[B // 2:], sig[:N - B // 2], atol=1e-12)


# ---------------------------------------------------------------------------
# Dominant frequency after pitch shift
# ---------------------------------------------------------------------------

def _dominant_freq(signal, fs):
    """Estimate pitch via median inter-zero-crossing period.

    More robust than FFT argmax for Hann-windowed granular output, where
    amplitude modulation at the grain rate biases the FFT peak and the
    Hilbert instantaneous frequency.  The median period across all rising
    zero-crossings discards the few crossings disturbed by grain wraps.
    """
    crossings = np.where((signal[:-1] <= 0) & (signal[1:] > 0))[0]
    if len(crossings) < 2:
        return np.nan
    return fs / float(np.median(np.diff(crossings)))


@pytest.mark.parametrize("semitones", [7, -7, 12, -12])
def test_dominant_freq_within_1_percent(semitones):
    """Dominant FFT frequency of a shifted 440 Hz sine must be within 1% of
    the expected ratio 440 * 2^(semitones/12)."""
    freq = 440.0
    sig = _sine(freq=freq, duration=2.0)
    out = pitch_shift(sig, FS, semitones=semitones)

    # Skip startup transient before measuring
    B = _buf_size()
    out_steady = out[B:]

    dominant = _dominant_freq(out_steady, FS)
    expected = freq * 2.0 ** (semitones / 12.0)

    assert abs(dominant - expected) / expected < 0.01, (
        f"semitones={semitones}: got {dominant:.2f} Hz, expected {expected:.2f} Hz"
    )
