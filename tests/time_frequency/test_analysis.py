import numpy as np

from python.time_frequency import spectrogram, stft

FS = 44100
FRAME = 512
HOP = 128


def _sine(freq=440.0, n=4096):
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * freq * t)


def test_spectrogram_shape():
    sig = _sine(n=4096)
    sp = spectrogram(sig, FRAME, HOP)
    S = stft(sig, FRAME, HOP)
    assert sp.shape == S.shape


def test_spectrogram_peak_bin_dominates():
    sig = _sine(n=4096)
    sp = spectrogram(sig, FRAME, HOP)
    assert np.max(sp) - np.min(sp) > 60.0


def test_spectrogram_silent_signal_all_negative():
    sig = np.zeros(4096)
    sp = spectrogram(sig, FRAME, HOP)
    assert np.all(sp < 0.0)
