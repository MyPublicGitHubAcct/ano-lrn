import numpy as np
import pytest

from python.physical import pluck

FS = 44100
DURATION = 2.0  # seconds — long enough for decay and FFT resolution


def _dominant_freq(signal, fs):
    """Return the frequency of the largest FFT bin, skipping the first 200 ms.

    The Karplus-Strong excitation (white noise) often has a large initial
    transient whose random spectral peaks can outweigh the fundamental before
    the loss filter has shaped the spectrum. Discarding the first 200 ms lets
    the resonance establish itself and gives a clean estimate.
    """
    start = int(0.2 * fs)
    seg = signal[start:]
    spectrum = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(len(seg), 1 / fs)
    return freqs[np.argmax(spectrum)]


def _harmonic_energy(signal, fs, fundamental, harmonics):
    """Sum power at the specified harmonic multiples (nearest FFT bin to each)."""
    spectrum = np.abs(np.fft.rfft(signal)) ** 2
    freqs = np.fft.rfftfreq(len(signal), 1 / fs)
    total = 0.0
    for k in harmonics:
        idx = np.argmin(np.abs(freqs - k * fundamental))
        total += spectrum[idx]
    return total


def test_pluck_returns_tuple_with_correct_shape():
    """pluck must return a (t, signal) tuple where both arrays have length fs * duration."""
    t, sig = pluck(freq=440.0, fs=FS, duration=DURATION)
    expected = int(FS * DURATION)
    assert t.shape == (expected,)
    assert sig.shape == (expected,)


def test_pluck_dominant_frequency_440():
    """Dominant FFT peak must be within 2 Hz of the target frequency (440 Hz)."""
    _, sig = pluck(freq=440.0, fs=FS, duration=DURATION)
    assert abs(_dominant_freq(sig, FS) - 440.0) < 2.0


def test_pluck_dominant_frequency_220():
    """Dominant FFT peak must be within 2 Hz of 220 Hz to confirm pitch scales correctly."""
    _, sig = pluck(freq=220.0, fs=FS, duration=DURATION)
    assert abs(_dominant_freq(sig, FS) - 220.0) < 2.0


def test_pluck_signal_decays_below_minus40db():
    """Signal must decay to below −40 dB of its peak RMS within the specified duration.

    The Karplus-Strong averaging filter is lossy (damping < 1), so the signal
    envelope must fall substantially over the output window.
    """
    _, sig = pluck(freq=440.0, fs=FS, duration=DURATION, damping=0.995)
    peak_rms = np.sqrt(np.mean(sig[:FS // 10] ** 2))  # RMS of first ~100 ms
    tail_rms = np.sqrt(np.mean(sig[-FS // 10:] ** 2))  # RMS of last ~100 ms
    # tail must be at least 40 dB quieter than the attack
    assert tail_rms < peak_rms * 10 ** (-40 / 20)


def test_pluck_pickup_mid_weakens_even_harmonics():
    """pickup=0.5 (string midpoint) must have weaker even harmonics than pickup=0.1.

    A pickup at the exact string midpoint coincides with a node of all even harmonics,
    so their energy is minimal — analogous to a guitar neck pickup vs. bridge pickup.
    """
    freq = 220.0
    _, sig_mid = pluck(freq=freq, fs=FS, duration=DURATION, pickup=0.5)
    _, sig_near = pluck(freq=freq, fs=FS, duration=DURATION, pickup=0.1)

    even_harmonics = [2, 4, 6, 8]
    energy_mid = _harmonic_energy(sig_mid, FS, freq, even_harmonics)
    energy_near = _harmonic_energy(sig_near, FS, freq, even_harmonics)

    assert energy_mid < energy_near


def test_pluck_output_is_finite():
    """Output must contain no NaN or Inf values for standard parameters."""
    _, sig = pluck(freq=330.0, fs=FS, duration=1.0)
    assert np.all(np.isfinite(sig))


def test_pluck_time_axis_starts_at_zero_and_matches_fs():
    """Time axis must start at 0 and have the correct sample spacing (1/fs)."""
    t, _ = pluck(freq=440.0, fs=FS, duration=0.5)
    assert t[0] == pytest.approx(0.0)
    assert (t[1] - t[0]) == pytest.approx(1.0 / FS)


def test_pluck_seed_reproducible():
    """The same seed must produce identical output; different seeds must differ."""
    t1, s1 = pluck(freq=440.0, fs=FS, duration=0.1, seed=7)
    t2, s2 = pluck(freq=440.0, fs=FS, duration=0.1, seed=7)
    t3, s3 = pluck(freq=440.0, fs=FS, duration=0.1, seed=99)
    np.testing.assert_array_equal(s1, s2)
    assert not np.array_equal(s1, s3)
