import numpy as np

from python.time_frequency import spectrogram, stft

FS = 44100
FRAME = 512
HOP = 128


def _sine(freq=440.0, n=4096):
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * freq * t)


def test_spectrogram_shape():
    """spectrogram must return a matrix with the same time-frequency dimensions as the underlying STFT."""
    sig = _sine(n=4096)
    sp = spectrogram(sig, FRAME, HOP)
    S = stft(sig, FRAME, HOP)
    assert sp.shape == S.shape


def test_spectrogram_peak_bin_dominates():
    """For a pure sine, the peak bin must stand at least 60 dB above the noise floor (confirms dB scaling)."""
    sig = _sine(n=4096)
    sp = spectrogram(sig, FRAME, HOP)
    assert np.max(sp) - np.min(sp) > 60.0


def test_spectrogram_silent_signal_all_negative():
    """A silent signal has zero magnitude; in dB (with small epsilon) all values must be negative."""
    sig = np.zeros(4096)
    sp = spectrogram(sig, FRAME, HOP)
    assert np.all(sp < 0.0)


def test_spectrogram_uses_amplitude_db_convention():
    """spectrogram must use 20·log10(|STFT|) (amplitude), not 10·log10(|STFT|²) (same result, but confirms intent)."""
    sig = _sine(n=4096)
    S = stft(sig, FRAME, HOP)
    sp = spectrogram(sig, FRAME, HOP)
    expected = 20.0 * np.log10(np.abs(S) + 1e-12)
    np.testing.assert_allclose(sp, expected, atol=1e-9)


def test_spectrogram_peak_bin_consistent_across_frames():
    """For a steady sine at f, the max-magnitude frequency bin must be at the expected FFT bin in every frame."""
    freq = 1000.0
    sig = _sine(freq=freq, n=4096)
    sp = spectrogram(sig, FRAME, HOP)
    freqs_ax = np.fft.rfftfreq(FRAME, d=1.0 / FS)
    expected_bin = np.argmin(np.abs(freqs_ax - freq))
    # For each steady-state frame (skip first few), peak bin must be at expected_bin
    for frame_idx in range(2, sp.shape[1]):
        peak_bin = np.argmax(sp[:, frame_idx])
        assert abs(peak_bin - expected_bin) <= 1

